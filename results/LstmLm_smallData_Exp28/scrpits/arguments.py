# -*- coding: utf-8 -*-
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='LanguageModel')
    
    
    
    parser.add_argument('--Nplot', default=100,type=int, help='number of batches between each plot')
        
    parser.add_argument('--dataset', default='smallData', type=str) 
    
    parser.add_argument('--dataPath', default="C://Users//simon//Desktop//HMS//projects//LM_word//data//splitted//smallData// ", type=str)
    
    parser.add_argument('--vis', default=True, type= bool) 
    
    parser.add_argument('--serverVisdom', default='http://localhost',type= str) 
    
    parser.add_argument('--portVisdom', default=8097,type= int) 
    
    parser.add_argument('--cuda', default=True,type= bool) 
    
    parser.add_argument("--devid", type=int, default=-1)
    parser.add_argument("--vocab_size", type=int, default=100)


    parser.add_argument("--model", choices=["NnLm", "LstmLm"], default="LstmLm")
    parser.add_argument("--nhid", type=int, default=10)
    parser.add_argument("--nlayers", type=int, default=1)

    parser.add_argument("--tieweights", default=True)
    parser.add_argument("--maxnorm", type=float, default=False)
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--epochs", type=int, default=19)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.25)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--bptt", type=int, default=25)
    parser.add_argument("--clip", type=float, default=5)

    # Adam parameters
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD parameters
    parser.add_argument("--mom", type=float, default=0.99)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", default=False)
    
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    if not args.cuda:
        print('*** WARNING: CUDA NOT ENABLED ***')

    return args

