# -*- coding: utf-8 -*-



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.autograd import Variable as V
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torchtext
from torch.autograd import Variable
from tqdm import tqdm
import random
from visualisation import visdom_plot
import itertools


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
        if args.maxnorm==False:
            maxNorm=None
        else:
            maxNorm=args.maxnorm
        self.lut = nn.Embedding(self.vsize, self.nhid, max_norm=maxNorm)
        self.rnn = nn.LSTM(
            input_size=self.nhid,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            dropout=self.dropout)
        self.drop = nn.Dropout(self.dropout)
        self.proj = nn.Linear(self.nhid, self.vsize)
        self.trainingEpochs=0 #number of epochs already trained
        self.trainingBatches=0
        self.temperature=1
        self.generationIteratorBuilt=False
        
        if self.tieweights:
            # See https://arxiv.org/abs/1608.05859
            # Seems to improve ppl by 13%.
            self.proj.weight = self.lut.weight
        
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, input, hid):
        emb = self.lut(input)
        hids, hid = self.rnn(emb, hid)
        # Detach hiddens to truncate the computational graph for BPTT.
        # Recall that hid = (h,c).
        return self.proj(self.drop(hids)), tuple(map(lambda x: x.detach(), hid))
    
    def next_N_words(self, word, hid_, TEXT, length_to_predict):
        output=[]
        
        for i in range(length_to_predict):
            wordIndex=TEXT.vocab.stoi[word]
            input_tensor=Variable(torch.LongTensor([[wordIndex]]))
            
            if torch.cuda.is_available():
                input_tensor=input_tensor.cuda()
            
            emb = self.lut(input_tensor)
            
            hids, hid_ = self.rnn(emb, hid_)
            
            next_logits=self.proj((hids)).view(-1)
            next_distr=torch.nn.Softmax(dim=0)(torch.mul(next_logits, 1/self.temperature))
            next_index=torch.multinomial(next_distr,1)
            
            next_word=TEXT.vocab.itos[next_index.cpu().data[0]]
            output+=[next_word]
            word=next_word
        return(output)

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)
    

    def train_epoch(self, iter, loss, optimizer, viz, win, TEXT, infoToPlot=None):
        self.train()
        self.trainingBatches=0

        train_loss = 0
        nwords = 0

        hid = None
        batch_id=0
        for batch in tqdm(iter):
            self.trainingBatches+=1
            
            if hid is not None:
                hid[0].detach_()
                hid[1].detach_()
                
            optimizer.zero_grad()
            x = batch.text
            y = batch.target
            
            if torch.cuda.is_available():
                x=x.cuda()
                y=y.cuda()
                
            out, hid = self(x, hid if hid is not None else None)
            bloss = loss(out.view(-1, self.vsize), y.view(-1))
            
            bloss.backward()
            train_loss += bloss
            # bytetensor.sum overflows, so cast to int
            local_nwords= y.ne(self.padidx).int().sum()
            nwords += local_nwords
            if self.args.clip > 0:
                clip_grad_norm_(self.parameters(), self.args.clip)

            optimizer.step()
            
            
            
            if not infoToPlot is None:
                infoToPlot['trainPerp']+=[np.exp(bloss.item()/local_nwords.item())]
                
            if batch_id % 100 == 0:
                sampled_sentences=self.generate_predictions(TEXT)
                #print(sampled_sentences)
                infoToPlot['generated']=sampled_sentences
                #print(infoToPlot['generated'])
                win = visdom_plot(viz, win, infoToPlot)
                self.train()


                
            
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
            
            if torch.cuda.is_available():
                x=x.cuda()
                y=y.cuda()
                
                
            out, hid = self(x, hid if hid is not None else None)
            valid_loss += loss(out.view(-1, self.vsize), y.view(-1))
            nwords += y.ne(self.padidx).int().sum()
            
        if not infoToPlot is None:
            infoToPlot['validPerp']+=[np.exp(valid_loss.data[0]/nwords.data[0])]
            win = visdom_plot(viz, win, infoToPlot, valid=True)
            
            
        return valid_loss.data[0], nwords.data[0]

    def generate_predictions(self, TEXT, outputDirectory=None, epoch=None, saveOutputs=False):
        if outputDirectory==None:
            outputDirectory=self.args.directoryData
        if epoch==None:
            epoch=self.trainingEpochs
        
            
        self.eval()
        batch_number=10
        outputs=[dict({'input_sentence':'','expected_sentence':'','output_sentence':''}) for _ in range(batch_number)]
        #prepare the data to generate on
        
        if not self.generationIteratorBuilt:
            data = torchtext.datasets.LanguageModelingDataset(
                path=os.path.join(os.getcwd(),"data", "splitted", "smallData","gen.txt"),
                text_field=TEXT)
            data_iter = torchtext.data.BPTTIterator(data, batch_number, 100, device=self.args.devid, train=False)
            #print()
            #print("Generating the next 80 words, from the 20 first ones")
            self.iterator=itertools.cycle(iter(data_iter))
            self.generationIteratorBuilt=True
        sample=next(self.iterator)
        input_sentence=sample.text[:20,:]
        if torch.cuda.is_available:
            input_sentence=input_sentence.cuda()
        expected_sentence=sample.text
        
        #we will give the 20 first words of a sentence, and predict the 80 next characters
        input_words=[[] for _ in range(batch_number)]        
        for i in range(batch_number):
                input_words[i]=[TEXT.vocab.itos[x] for x in input_sentence[:,i].data.tolist()]
                #print(input_words)
        
        expected_output_words=[[] for _ in range(batch_number)]        
        for i in range(batch_number):
                expected_output_words[i]=[TEXT.vocab.itos[x] for x in expected_sentence[:,i].data.tolist()]
                #print(output_words)          
      
        
       
        if saveOutputs:
            with open(os.path.join(outputDirectory,self.__class__.__name__ + ".preds.txt"), "a") as f:
                f.write('*' * 20)
                f.write('\n \n NEW : EPOCH {} \n \n '.format(epoch))
                f.write('*' * 20)
                f.write('\n')
            
        #first we run the model on the first 20 words, in order to give context to the hidden state
        output_sentence, hidden = self(input_sentence, None)
            
        for batch_index in range(batch_number):          
            
            #then we run the model using the  computed hidden states
            input_word=expected_output_words[batch_index][20]
            
            #we select the correct hidden state for the batch
            #because next_N_words take a single word as input
            local_h=hidden[0][:,batch_index:batch_index+1,:]
            local_c=hidden[1][:,batch_index:batch_index+1,:]
            local_hidden=(local_h, local_c)
            
            output=self.next_N_words(input_word, local_hidden, TEXT, 80)

            outputs[batch_index]['input_sentence']=''.join([x+' ' for x in input_words[batch_index]])
            outputs[batch_index]['expected_sentence']=''.join([x+' ' for x in expected_output_words[batch_index]])
            outputs[batch_index]['output_sentence']=''.join([x+' ' for x in output])
            
            if saveOutputs:
                with open(os.path.join(outputDirectory,self.__class__.__name__ + ".preds.txt"), "a") as f:
    
                    f.write('input sentence : \n')
                    f.write( ''.join([x+' ' for x in input_words[batch_index]]))
                    f.write('\n \n')
                    
                    f.write('expected output sentence : \n')
                    f.write(''.join([x+' ' for x in expected_output_words[batch_index]]))
                    f.write('\n \n')
                    
                    f.write('sampled output sentence : \n')
                    f.write(''.join([x+' ' for x in output]))
                    f.write('\n \n \n ')
    
 
        return(outputs)
            
                
        


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
    
    


