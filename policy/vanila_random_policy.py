from headers import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class VanilaRandomPolicy(torch.nn.Module):
    def __init__(self, D_out):
        """
        D_out: a int or a list of ints in length of degree of freedoms
        hiddens, kernel_sizes, strides: either an int or a list of ints with the same length
        """
        super(VanilaRandomPolicy, self).__init__()
        if isinstance(D_out, int):
            self.D_out = [D_out]
        else:
            self.D_out = D_out
        self.out_dim = sum(D_out)

    def forward(self, x, gumbel_noise = 1.0):
        """
        compute the forward pass of the model.
        return logits and the softmax prob w./w.o. gumbel noise
        """
        action = []
        for d in self.D_out:
            u = Variable(torch.rand(x.size(0), d)).type(FloatTensor)
            action.append(F.softmax(u))
            #action.append(u)
        return action
