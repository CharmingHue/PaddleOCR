from timm.models.layers import DropPath
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



class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            MemoryEfficientSwish(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                            #nn.Identity()
                         )
    def forward(self, x):
        return self.act_block(x)
    
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
