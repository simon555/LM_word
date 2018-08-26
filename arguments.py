# -*- coding: utf-8 -*-
import argparse

import torch
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(description='LanguageModel')
    
    
    
   
    if os.name=='nt':
        parser.add_argument('--dataset', default='smallData', type=str) 
    else:
        parser.add_argument('--dataset', default='springer_cui_tokenized', type=str) 
    
    
    parser.add_argument('--dataPath', default="C://Users//simon//Desktop//HMS//projects//LM_word//data//splitted//smallData// ", type=str)
    
    parser.add_argument('--vis', default=True, type= str2bool) 
    
    parser.add_argument('--serverVisdom', default='http://localhost',type= str) 
    
    parser.add_argument('--portVisdom', default=8097,type= int) 
    
    parser.add_argument('--cuda', default=True,type= str2bool)
    
    parser.add_argument('--scheduleLearningRate', default=True, type= str2bool) 
    
    
    if os.name=='nt':
        parser.add_argument("--devid", type=int, default=0)
        parser.add_argument("--nhid", type=int, default=256)
        parser.add_argument("--nlayers", type=int, default=1)
        parser.add_argument('--Nplot', default=300,type=int, help='number of batches between each plot')
        parser.add_argument('--Nsave', default=300,type=int, help='number of batches between each model save')

    
    else:
        parser.add_argument("--devid", type=int, default=6)
        parser.add_argument("--nhid", type=int, default=512)
        parser.add_argument("--nlayers", type=int, default=2)
        parser.add_argument('--Nplot', default=500,type=int, help='number of batches between each plot')
        parser.add_argument('--Nsave', default=5000,type=int, help='number of batches between each model save')

    

    parser.add_argument("--vocab_size", type=int, default=-1, help='size of the vocab, -1 means full size')
    
    parser.add_argument("--grainLevel", type=str, default='word')



    parser.add_argument("--model", choices=["adaptive_lstm", "lstm"], default="lstm")
    parser.add_argument("--loss", choices=["regularCrossEntropy", "adaptive"], default="adaptive")

    
    
    

    parser.add_argument("--tieweights", default=True)
    parser.add_argument("--maxnorm", type=float, default=False)
    parser.add_argument("--dropout_LSTM", type=float, default=0.4)
    parser.add_argument("--dropout_linear", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=100000)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.25)
    parser.add_argument("--wd", type=float, default=1e-6)

    parser.add_argument("--bsz", type=int, default=64)
    parser.add_argument("--bptt", type=int, default=20)
    parser.add_argument("--clip", type=float, default=5)
    
    #generator parameters
    parser.add_argument("--gen_bsz", type=int, default=5)
    parser.add_argument("--gen_bptt", type=int, default=100)
    parser.add_argument("--gen_warmup", type=int, default=20)

    # Adam parameters
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    # SGD parameters
    parser.add_argument("--mom", type=float, default=0.99)
    parser.add_argument("--dm", type=float, default=0)
    parser.add_argument("--nonag", default=False)
    
    parser.add_argument("--lazyLoading", type=str2bool,  default=True)

    
    args = parser.parse_args()

    args.cuda = args.cuda and torch.cuda.is_available()

    if not args.cuda:
        print('*** WARNING: CUDA NOT ENABLED ***')

    return args

