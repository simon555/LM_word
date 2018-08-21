# -*- coding: utf-8 -*-

import torch.nn as nn
import torch



class Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.vsize=kwargs['vsize']
        self.padidx=kwargs['padidx']
        self.unkidx=kwargs['unkidx']
        # We do not want to give the model credit for predicting padding symbols,
        # this can decrease ppl a few points.
        
        weight = torch.FloatTensor(self.vsize).fill_(1)
        weight[self.padidx] = 0
        weight[self.unkidx] = 0
        
        self.loss=nn.NLLLoss(weight= weight, reduction='sum')
        
        
    def forward(self, input, target):
        return(self.loss(input, target))
        