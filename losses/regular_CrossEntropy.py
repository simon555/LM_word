# -*- coding: utf-8 -*-

import torch.nn as nn
import torch



class Loss(nn.Module):
    def __init__(self, **kwargs):
        super(Loss, self).__init__()
        self.vsize=kwargs['vsize']
        self.padidx=kwargs['padidx']
        self.size_average=kwargs['size_average']
        # We do not want to give the model credit for predicting padding symbols,
        # this can decrease ppl a few points.
        
        weight = torch.FloatTensor(self.vsize).fill_(1)
        weight[self.padidx] = 0
        
        self.loss=nn.CrossEntropyLoss(weight= weight, size_average=self.size_average)
        
        
    def forward(self, input, target):
        return(self.loss(input, target))
        