import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


eposilo = 1e-6
dim = 123

dim = 39

dim = 128


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self,x):
        return x

class Neg(nn.Module):
    def __init__(self):
        super(Neg, self).__init__()
    def forward(self,x):
        return -x


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
    def forward(self,x):
        return torch.abs(x)

class Square(nn.Module):
    def __init__(self):
        super(Square, self).__init__()
    def forward(self,x):
        return torch.square(x)

class Sqrt(nn.Module):
    def __init__(self):
        super(Sqrt, self).__init__()
    def forward(self,x):
        return torch.sign(x)*torch.sqrt(torch.abs(x)+eposilo)

class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.activation = nn.Tanh()
    def forward(self,x):
        return self.activation(x)


class Exp(nn.Module):
    def __init__(self):
        super(Exp, self).__init__()
    def forward(self,x):
        return torch.exp(x)

class Inv(nn.Module):
    def __init__(self):
        super(Inv, self).__init__()
    def forward(self,x):
        return 1/(x+eposilo)

class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.activation = nn.Sigmoid()
    def forward(self,x):
        return self.activation(x)


class Softplus(nn.Module):
    def __init__(self):
        super(Softplus, self).__init__()
        self.activation = nn.Softplus()
    def forward(self,x):
        return self.activation(x)



#  †Aggregation Operator
class Sum(nn.Module):
    def __init__(self):
        super(Sum, self).__init__()
    def forward(self,x):
        assert x.shape[1]!=1
        return torch.unsqueeze(torch.sum(x,dim=-1),-1 ) # output [batch,1]

class Mean(nn.Module):
    def __init__(self):
        super(Mean, self).__init__()
    def forward(self,x):
        assert x.shape[1]!=1
        return torch.unsqueeze(torch.mean(x,dim=-1),-1 ) # output [batch,1]

class FFN(nn.Module):
    def __init__(self):
        super(FFN, self).__init__()
        self.FC = nn.Linear(dim,1)
    def forward(self,x):
        assert x.shape[1]!=1
        return self.FC(x)  # output [batch,1]

class FFN_D(nn.Module):
    def __init__(self):
        super(FFN_D, self).__init__()
        self.FC = nn.Linear(dim,dim)
    def forward(self,x):
        # assert x.shape[1]!=1
        return self.FC(x) 




# Element-wise Operator
class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
    def forward(self,x,y):
        return x+y

class Mul(nn.Module):
    def __init__(self):
        super(Mul, self).__init__()
    def forward(self,x,y):
        return torch.multiply(x,y)

class ConcatLinear(nn.Module):
    def __init__(self):
        super(ConcatLinear, self).__init__()
        # self.FC = nn.Linear(256,1)
        self.FC = nn.Linear(2*dim,dim)
    def forward(self,x,y):
        assert x.shape==y.shape
        assert x.shape[1]!=1
        out = torch.cat([x,y],dim=-1)
        return self.FC(out)

# -----------------Additional------------------------
class Log(nn.Module):
    def __init__(self):
        super(Log, self).__init__()
    def forward(self,x):
        return torch.sign(x)*torch.log(torch.abs(x)+eposilo)





MixedOp = nn.ModuleList([])
MixedOp.extend([Neg()]) # 0
MixedOp.extend([Abs()]) # 1
MixedOp.extend([Square()]) # 2
MixedOp.extend([Sqrt()]) # 3

MixedOp.extend([Tanh()]) # 4
MixedOp.extend([Sigmoid()]) # 5
MixedOp.extend([FFN()]) # 6

MixedOp.extend([Add()]) # 7
MixedOp.extend([Mul()]) # 8

#----------------------------------------------------------------------
MixedOp1 = nn.ModuleList([])
MixedOp1.extend([Neg()]) # 0
MixedOp1.extend([Abs()]) # 1
MixedOp1.extend([Square()]) # 2
MixedOp1.extend([Sqrt()])  # 3
MixedOp1.extend([Tanh()])  # 4
MixedOp1.extend([Exp()])   # 5
MixedOp1.extend([Inv()])   # 6
MixedOp1.extend([Sigmoid()]) # 7
MixedOp1.extend([Softplus()]) # 8


MixedOp1.extend([Sum()]) # 9
MixedOp1.extend([FFN()]) # 10

MixedOp1.extend([Add()]) # 11
MixedOp1.extend([Mul()]) # 12
MixedOp1.extend([ConcatLinear()]) #13
#-------------------------------------------------------

NAS_OPS = {
    'neg': lambda x: Neg(),
    'abs': lambda x: Abs(),
    'square': lambda x: Square(),
    'sqrt': lambda x: Sqrt(),
    'tanh': lambda x: Tanh(),
    'exp': lambda x: Exp(),
    'inv': lambda x: Inv(),
    'sigmoid': lambda x: Sigmoid(),
    'softplus': lambda x: Softplus(),


    'sum': lambda x: Sum(),
    'ffn': lambda x: FFN(),


    'add': lambda x: Add(),
    'mul': lambda x: Mul(),
    'concat': lambda x: ConcatLinear(),

    'log': lambda x: Log(),
    'mean': lambda x: Mean(),
    'ffn_d':lambda x: FFN_D(),#14
}
