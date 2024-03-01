from typing import List

import math
from functools import partial

import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.autograd import PyLayer
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
import os

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

'''
#* >>>>>svtr Patchembeding start<<<<<
'''
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[32, 100],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=2,
                 patch_size=[4, 4],
                 mode='pope'):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
        elif mode == 'linear':
            self.proj = nn.Conv2D(
                1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x
'''
>>>>>svtr Patchembeding end<<<<<
'''
    
    
'''
#* >>>>>cloFormer Patchembeding start<<<<<
'''
class BasicBlock(nn.Layer):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
                        nn.Conv2D(in_channels, out_channels, kernel_size, stride, kernel_size//2),
                        nn.SyncBatchNorm(out_channels),
                        nn.ReLU()
                    )
    
    def forward(self, x: paddle.Tensor):
        return self.conv(x)


class PatchEmbedding(nn.Layer):

    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        self.conv1 = BasicBlock(in_channels, out_channels//2, 3, 2)
        self.conv2 = BasicBlock(out_channels//2, out_channels, 3, 2)
        self.conv3 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv4 = BasicBlock(out_channels, out_channels, 3, 1)
        self.conv5 = nn.Conv2D(out_channels, out_channels, 1, 1, 0)
        self.layernorm = nn.GroupNorm(1, out_channels)

    def forward(self, x: paddle.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return self.layernorm(x)
'''
>>>>>cloFormer Patchembeding end<<<<<
'''


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
  
        b, c, h, w = x.shape
        qkv = to_qkv(x) #(b (3 m d) h w)
        qkv = mixer(qkv).reshape((b, 3, -1, h, w)).transpose(perm=[1, 0, 2, 3, 4]) #(3 b (m d) h w)
        q, k, v = qkv #(b (m d) h w)
        attn = attn_block(q.multiply(k)).multiply(paddle.to_tensor(self.scalor))
        attn = self.attn_drop(paddle.tanh(attn))
        res = attn.multiply(v) #(b (m d) h w)
        return res
        
    def low_fre_attention(self, x : paddle.Tensor, to_q: nn.layer, to_kv: nn.Layer, avgpool: nn.Layer):
        '''
        x: (b c h w)
        '''
        
        b, c, h, w = x.shape
        
        q = to_q(x).reshape((b, -1, self.dim_head, h*w)).transpose([0, 1, 3, 2]) #(b m (h w) d)
        kv = avgpool(x) #(b c h w)
        kv = to_kv(kv).reshape((b, 2, -1, self.dim_head, (h*w)//(self.window_size**2))).transpose([1, 0, 2, 4, 3]) #(2 b m (H W) d)
        k, v = kv #(b m (H W) d)
        attn = self.scalor * q @ k.transpose([0, 1, 3, 2]) #(b m (h w) (H W))
        attn = self.attn_drop(nn.functional.softmax(attn))
        res = attn @ v #(b m (h w) d)
        res = res.transpose([0, 1, 3, 2]).reshape((b, -1, h, w))
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
        return self.proj_drop(self.proj(paddle.concat(res, axis=1)))


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
                                nn.Conv2D(dim, dim, mlp_kernel_size, (2,1), mlp_kernel_size//2),
                                nn.SyncBatchNorm(dim),
                                nn.Conv2D(dim, out_dim, 1, 1, 0),
                            )
        self.mlp = ConvFFN(dim, mlp_hidden_dim, mlp_kernel_size, (stride, 1), out_dim, 
                        drop_out=mlp_drop)
    def forward(self, x: paddle.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = self.downsample(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


'''
#* >>>>>build CloLayer start<<<<<
'''
class CloLayer(nn.Layer):

    def __init__(self,
                 depth,
                 dim,
                 out_dim,
                 num_heads,
                 group_split: List[int],
                 kernel_sizes: List[int],
                 window_size: int,
                 mlp_kernel_size: int,
                 mlp_ratio: int,
                 attn_drop=0,
                 mlp_drop=0.,
                 qkv_bias=True,
                 drop_paths=[0., 0.],
                 downsample=True,):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.LayerList(
                        [
                            EfficientBlock(dim, dim, num_heads, group_split, kernel_sizes, window_size,
                                mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[i])
                                for i in range(depth-1)
                        ]
                    )
        if downsample is True:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 2, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))
        else:
            self.blocks.append(EfficientBlock(dim, out_dim, num_heads, group_split, kernel_sizes, window_size,
                            mlp_kernel_size, mlp_ratio, 1, attn_drop, mlp_drop, qkv_bias, drop_paths[-1]))

    def forward(self, x: paddle.Tensor):
        for blk in self.blocks:
            x = blk(x)
        return x
'''
>>>>>build CloLayer end<<<<<
'''
 
'''
#* >>>>>build CloFormerNet end<<<<<
'''    
class CloFormerNet(nn.Layer):

    def __init__(
                self,
                img_size=[32, 100],
                in_chans=3,
                embed_dims=[32, 64, 128, 256],
                depths=[2, 2, 6, 2],
                num_heads=[4, 4, 8, 16],
                group_splits=[[3, 1], [2, 2], [4, 4], [4, 12]],
                kernel_sizes=[[3], [5], [7], [9]],
                window_sizes=[8, 4, 2, 1],
                mlp_kernel_sizes=[5, 5, 5, 5],
                mlp_ratios=[4, 4, 4, 4],
                attn_drop=0.,
                mlp_drop=0.,
                last_drop=0.1,
                drop_rate=0.,
                qkv_bias=True,
                drop_path_rate=0.1,
                patch_type='clo',
                out_channels=192,
                out_char_num=25,
                use_lenhead=False,
                last_stage=True,
                prenorm=True,
                norm_layer='nn.LayerNorm',
                epsilon=1e-6,
                
                **kwargs                 
                 ):
        super().__init__()
        self.num_layers = len(depths)
        self.mlp_ratios = mlp_ratios
        self.patch_type = patch_type
        self.last_stage = last_stage
        self.out_channels = out_channels
        self.use_lenhead = use_lenhead
        self.prenorm = prenorm
        
        
        if self.patch_type == "svtr":
            self.patch_embed = PatchEmbed(embed_dim=embed_dims[0])
            self.pos_embed = self.create_parameter(shape=[1, embed_dims[0], img_size[0]//4, img_size[1]//4],
                                                   default_initializer=zeros_)
            self.add_parameter("pos_embed", self.pos_embed)
            self.pos_drop = nn.Dropout(p=drop_rate)
            trunc_normal_(self.pos_embed)
            self.apply(self._init_weights)
        else:
            self.patch_embed = PatchEmbedding(in_chans, embed_dims[0])


        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            if i_layer != self.num_layers-1:
                layer = CloLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer+1], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer], 
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop, 
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], True, )
            else:
                layer = CloLayer(depths[i_layer], embed_dims[i_layer], embed_dims[i_layer], num_heads[i_layer],
                            group_splits[i_layer], kernel_sizes[i_layer], window_sizes[i_layer], 
                            mlp_kernel_sizes[i_layer], mlp_ratios[i_layer], attn_drop, mlp_drop, 
                            qkv_bias, dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])], False, )
            self.layers.append(layer)
        
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2D([1, out_char_num])
            self.last_conv = nn.Conv2D(
                in_channels=embed_dims[3],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dims[-1], epsilon=epsilon)
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dims[3], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(
                p=last_drop, mode="downscale_in_infer")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
            
            
    def forward_feature(self, x):
        '''
        x: (b 3 h w)
        '''
        x = self.patch_embed(x)
        if self.patch_type == "svtr":
            x = x + self.pos_embed
            x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)           
        if not self.prenorm:
            x = self.norm(x)           
        return x

    def forward(self, x):
        x = self.forward_feature(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        if self.use_lenhead:
            return x, len_x
        return x
'''
>>>>>build CloFormerNet end<<<<<
'''
if __name__ == "__main__":
    input = paddle.randn([1, 3, 32, 100])
    model = CloFormerNet(patch_type='clo')
    output = model(input)
    print(model)
    print(output.shape)