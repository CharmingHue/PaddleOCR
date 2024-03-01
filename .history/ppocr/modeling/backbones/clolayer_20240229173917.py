from typing import List

import math
from functools import partial

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer

##Swish激活函数  由之前的激活函数复合而成出来的   
##通过创建 PyLayer 子类的方式实现Python端自定义算子
class SwishImplementation(PyLayer):
    def forward(ctx, i):
        result = i * F.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = F.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Layer):
    def forward(self, x):
        return SwishImplementation.apply(x)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'



class AttnMap(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2D(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2D(dim, dim, 1, 1, 0)
                            #nn.Identity()
                         )
    def forward(self, x):
        return self.act_block(x)
    
class EfficientAttention(nn.Layer):

    def __init__(self, dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size=7, 
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split
        convs = []
        act_blocks = []
        qkvs = []
        #projs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2D(3*self.dim_head*group_head, 3*self.dim_head*group_head, kernel_size,
                         1, kernel_size//2, groups=3*self.dim_head*group_head))
            act_blocks.append(AttnMap(self.dim_head*group_head))
            qkvs.append(nn.Conv2D(dim, 3*group_head*self.dim_head, 1, 1, 0, bias_attr=qkv_bias))
            #projs.append(nn.Linear(group_head*self.dim_head, group_head*self.dim_head, bias=qkv_bias))
        if group_split[-1] != 0:
            self.global_q = nn.Conv2D(dim, group_split[-1]*self.dim_head, 1, 1, 0, bias_attr=qkv_bias)
            self.global_kv = nn.Conv2D(dim, group_split[-1]*self.dim_head*2, 1, 1, 0, bias_attr=qkv_bias)
            #self.global_proj = nn.Linear(group_split[-1]*self.dim_head, group_split[-1]*self.dim_head, bias=qkv_bias)
            self.avgpool = nn.AvgPool2D(window_size, window_size) if window_size!=1 else nn.Identity()

        self.convs = nn.LayerList(convs)
        self.act_blocks = nn.LayerList(act_blocks)
        self.qkvs = nn.LayerList(qkvs)
        self.proj = nn.Conv2D(dim, dim, 1, 1, 0, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def high_fre_attntion(self, x: paddle.Tensor, to_qkv: nn.Layer, mixer: nn.Layer, attn_block: nn.Layer):
        '''
        x: (b c h w)
        '''
        print(x.size)
        b, c, h, w = x.shape()[:]
        qkv = to_qkv(x) #(b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous() #(3 b (m d) h w)
        q, k, v = qkv #(b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(paddle.tanh(attn))
        res = attn.mul(v) #(b (m d) h w)
        return res
        
    def low_fre_attention(self, x : paddle.Tensor, to_q: nn.layer, to_kv: nn.Layer, avgpool: nn.Layer):
        '''
        x: (b c h w)
        '''
        
        b, c, h, w = x.shape()[:]
        
        q = to_q(x).reshape(b, -1, self.dim_head, h*w).transpose(-1, -2).contiguous() #(b m (h w) d)
        kv = avgpool(x) #(b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h*w)//(self.window_size**2)).permute(1, 0, 2, 4, 3).contiguous() #(2 b m (H W) d)
        k, v = kv #(b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2) #(b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v #(b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    def forward(self, x: paddle.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(paddle.cat(res, dim=1)))


class ConvFFN(nn.Layer):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv2D(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.dwconv = nn.Conv2D(hidden_channels, hidden_channels, kernel_size, stride, 
                                kernel_size//2, groups=hidden_channels)
        self.fc2 = nn.Conv2D(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: paddle.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class EfficientBlock(nn.Layer):

    def __init__(self, dim, out_dim, num_heads, group_split: List[int], kernel_sizes: List[int], window_size: int,
                 mlp_kernel_size: int, mlp_ratio: int, stride: int, attn_drop=0., mlp_drop=0., qkv_bias=True,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = EfficientAttention(dim, num_heads, group_split, kernel_sizes, window_size,
                                       attn_drop, mlp_drop, qkv_bias)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.stride = stride
        if stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                                nn.Conv2D(dim, dim, mlp_kernel_size, 2, mlp_kernel_size//2),
                                nn.SyncBatchNorm(dim),
                                nn.Conv2D(dim, out_dim, 1, 1, 0),
                            )
        self.mlp = ConvFFN(dim, mlp_hidden_dim, mlp_kernel_size, stride, out_dim, 
                        drop_out=mlp_drop)
    def forward(self, x: paddle.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.downsample(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x

if __name__ == '__main__':
    input = paddle.randn((4, 96, 56, 56),dtype="float32")
    model = EfficientBlock(96, 192, 3, [1, 1, 1], [7, 5], 7, 7, 4, 2)
    print(model(input).size())