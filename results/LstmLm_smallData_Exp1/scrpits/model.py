# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable as V
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm
import numpy as np
import torchtext

from tqdm import tqdm
import random
from visualisation import visdom_plot



random.seed(1111)
torch.manual_seed(1111)


class LstmLm(nn.Module):
    def __init__(self, args, vocab, padidx):
        super(LstmLm, self).__init__()

        self.args=args
        self.nhid=args.nhid
        self.nlayers=args.nlayers
        self.dropout=args.dropout
        self.tieweights=args.tieweights
        self.vsize = len(vocab.itos)
        self.padidx=padidx
        self.lut = nn.Embedding(self.vsize, self.nhid, max_norm=args.maxnorm)
        self.rnn = nn.LSTM(
            input_size=self.nhid,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            dropout=self.dropout)
        self.drop = nn.Dropout(self.dropout)
        self.proj = nn.Linear(self.nhid, self.vsize)
        
        if self.tieweights:
            # See https://arxiv.org/abs/1608.05859
            # Seems to improve ppl by 13%.
            self.proj.weight = self.lut.weight

    def forward(self, input, hid):
        emb = self.lut(input)
        hids, hid = self.rnn(emb, hid)
        # Detach hiddens to truncate the computational graph for BPTT.
        # Recall that hid = (h,c).
        return self.proj(self.drop(hids)), tuple(map(lambda x: x.detach(), hid))

    def train_epoch(self, iter, loss, optimizer, viz, win, infoToPlot=None):
        self.train()

        train_loss = 0
        nwords = 0

        hid = None
        batch_id=0
        for batch in tqdm(iter):
            optimizer.zero_grad()
            x = batch.text
            y = batch.target
            out, hid = self(x, hid if hid is not None else None)
            bloss = loss(out.view(-1, self.vsize), y.view(-1))
        
            bloss.backward()
            train_loss += bloss
            # bytetensor.sum overflows, so cast to int
            local_nwords= y.ne(self.padidx).int().sum()
            nwords += local_nwords
            if self.args.clip > 0:
                clip_grad_norm(self.parameters(), self.args.clip)

            optimizer.step()
            
            if not infoToPlot is None:
                infoToPlot['trainPerp']+=[np.exp(bloss.data[0]/local_nwords.data[0])]
                
            if batch_id % 100 == 0:
                win = visdom_plot(viz, win, infoToPlot)

                
            
            batch_id+=1
        return train_loss.data[0], nwords.data[0]

    def validate(self, iter, loss, viz, win, infoToPlot=None):
        self.eval()

        valid_loss = 0
        nwords = 0

        hid = None
        for batch in iter:
            x = batch.text
            y = batch.target
            out, hid = self(x, hid if hid is not None else None)
            valid_loss += loss(out.view(-1, self.vsize), y.view(-1))
            nwords += y.ne(self.padidx).int().sum()
            
        if not infoToPlot is None:
            infoToPlot['validPerp']+=[np.exp(valid_loss.data[0]/nwords.data[0])]
            win = visdom_plot(viz, win, infoToPlot, valid=True)
            
            
        return valid_loss.data[0], nwords.data[0]

    def generate_predictions(self, TEXT):
        self.eval()
        data = torchtext.datasets.LanguageModelingDataset(
            path="./data/splitted/smallData/gen.txt",
            text_field=TEXT)
        data_iter = torchtext.data.BPTTIterator(data, 211, 12, device=self.args.devid, train=False)
        outputs = [[] for _ in range(211)]
        print()
        print("Generating Kaggle predictions")
        for batch in tqdm(data_iter):
            # T x N x V
            scores, idxs = self(batch.text, None)[0][-3].topk(20, dim=-1)
            for i in range(idxs.size(0)):
                outputs[i].append([TEXT.vocab.itos[x] for x in idxs[i].data.tolist()])
        with open(self.__class__.__name__ + ".preds.txt", "w") as f:
            f.write("id,word\n")
            ok = 1
            for sentences in outputs:
                f.write("\n".join(["{},{}".format(ok+i, " ".join(x)) for i, x in enumerate(sentences)]))
                f.write("\n")
                ok += len(sentences)
