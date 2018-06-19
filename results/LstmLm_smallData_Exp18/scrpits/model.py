# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.autograd import Variable as V
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm
import numpy as np
import torchtext
from torch.autograd import Variable
from tqdm import tqdm
import random
from visualisation import visdom_plot

#from local_models.log_uniform.log_uniform import LogUniformSampler

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
        self.trainingEpochs=0 #number of epochs already trained
        
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
        self.trainingEpochs+=1
        return train_loss.data[0], nwords.data[0]

    def validate(self, iter, loss, viz=None, win=None, infoToPlot=None):
        self.eval()

        valid_loss = 0
        nwords = 0

        hid = None
        for batch in tqdm(iter):
            x = batch.text
            y = batch.target
            out, hid = self(x, hid if hid is not None else None)
            valid_loss += loss(out.view(-1, self.vsize), y.view(-1))
            nwords += y.ne(self.padidx).int().sum()
            
        if not infoToPlot is None:
            infoToPlot['validPerp']+=[np.exp(valid_loss.data[0]/nwords.data[0])]
            win = visdom_plot(viz, win, infoToPlot, valid=True)
            
            
        return valid_loss.data[0], nwords.data[0]

    def generate_predictions(self, TEXT, outputDirectory=None, epoch=None):
        if outputDirectory==None:
            outputDirectory=self.args.directoryData
        if epoch==None:
            epoch=self.trainingEpochs
            
        self.eval()
        data = torchtext.datasets.LanguageModelingDataset(
            path=os.path.join(os.getcwd(),"data", "splitted", "smallData","gen.txt"),
            text_field=TEXT)
        batch_number=10
        data_iter = torchtext.data.BPTTIterator(data, batch_number, 100, device=self.args.devid, train=False)
        outputs = [[] for _ in range(batch_number)]
        print()
        print("Generating Kaggle predictions")
        sample=next(iter(data_iter))
        input_sentence=sample.text
        expected_sentence=sample.target
        
        input_words=[[] for _ in range(batch_number)]        
        for i in range(batch_number):
                input_words[i]=[TEXT.vocab.itos[x] for x in input_sentence[:,i].data.tolist()]
                print(input_words)
        
        output_words=[[] for _ in range(batch_number)]        
        for i in range(batch_number):
                output_words[i]=[TEXT.vocab.itos[x] for x in expected_sentence[:,i].data.tolist()]
                print(output_words)
                
        # T x Nbatch 
        scores, idxs = self(input_sentence, None)[0].topk(1, dim=-1)
        idxs=idxs[:,:,0]
        for i in range(batch_number):
                outputs[i]=[TEXT.vocab.itos[x] for x in idxs[:,i].data.tolist()]
                print(outputs)
        
        with open(os.path.join(outputDirectory,self.__class__.__name__ + ".preds.txt"), "a") as f:
            for batch_index in range(batch_number):
                f.write('*' * 20)
                f.write('\n \n NEW EPOCH : {} \n \n '.format(epoch))
                f.write('*' * 20)
                
                f.write('input sentence : \n')
                f.write( ''.join([x+' ' for x in input_words[batch_index]]))
                f.write('\n')
                
                f.write('expected output sentence : \n')
                f.write(''.join([x+' ' for x in output_words[batch_index]]))
                f.write('\n')
                
                f.write('sampled output sentence : \n')
                f.write(''.join([x+' ' for x in outputs[batch_index]]))
                f.write('\n \n')
                
            
                
        


class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, decoding_module):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = decoding_module

    def forward(self, inputs, labels):
        if self.training:
            return self.sampled(inputs, labels, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, remove_accidental_match=False):
        batch_size, d = inputs.size()
        labels_ndarray = labels.data.cpu().numpy()
        sample_ids, true_freq, sample_freq = self.sampler.sample(self.nsampled, labels_ndarray)

        # gather true labels and sample ids
        true_weights = self.params.weight.index_select(0, labels)
        sample_weights = self.params.weight[sample_ids, :]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1)
        sample_logits = torch.matmul(inputs, torch.t(sample_weights))
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels_ndarray, sample_ids)
            if len(acc_hits) > 0:
                acc_hits = list(zip(*acc_hits))
                sample_logits[acc_hits] = -1e37

        # perform correction
        true_freq = Variable(type(inputs.data)(true_freq))
        sample_freq = Variable(type(inputs.data)(sample_freq))

        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = Variable(type(labels.data)(batch_size).zero_())
        return logits, new_targets

    def full(self, inputs, labels):
        return self.params(inputs), labels
    
    



