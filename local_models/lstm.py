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
import time
import pickle
from infoToTrack import updateDataStorage

#from local_models.log_uniform.log_uniform import LogUniformSampler

random.seed(1111)
torch.manual_seed(1111)


    


class LstmLm(nn.Module):
    def __init__(self, args, vocab, padidx):
        super(LstmLm, self).__init__()

        self.args=args
        self.nhid=args.nhid
        self.nlayers=args.nlayers
        self.dropout_LSTM=args.dropout_LSTM
        self.dropout_linear=args.dropout_linear
        self.tieweights=args.tieweights
        self.vsize = len(vocab.itos)
        self.padidx=padidx
        
        #number of batches of training seen
        self.batch_id=1
        
        if args.maxnorm==False:
            maxNorm=None
        else:
            maxNorm=args.maxnorm
            
        self.lut = nn.Embedding(self.vsize, self.nhid, max_norm=maxNorm)
        self.rnn = nn.LSTM(
            input_size=self.nhid,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            dropout=self.dropout_LSTM)
        self.drop = nn.Dropout(self.dropout_linear)
        self.trainingEpochs=0 #number of epochs already trained
        self.trainingBatches=0
        self.temperature=1
        self.generationIteratorBuilt=False
        
        if self.args.loss == 'regularCrossEntropy':
            self.lastLayer = nn.Linear(self.nhid, self.vsize)
            self.get_logProb = nn.LogSoftmax()
        elif self.args.loss =='adaptive':
            self.cutoffs=[int(self.vsize*0.2),int(self.vsize*0.5),int(self.vsize*0.7), int(self.vsize*0.9)]
            self.lastLayer=nn.AdaptiveLogSoftmaxWithLoss(self.nhid, self.vsize, self.cutoffs)

        
        
        
        self.best_ppl=np.inf
        
        if self.tieweights:
            # See https://arxiv.org/abs/1608.05859
            # Seems to improve ppl by 13%.
            if self.args.loss == 'regularCrossEntropy':
                self.lastLayer.weight = self.lut.weight
            elif self.args.loss == 'adaptive':
                print('not implemented tiedweights yet....')
        
        if torch.cuda.is_available():
            self.cuda(self.args.devid)
            
    
    
            
    
    
    def idx2word(self, indexes, TEXT):
        """
        takes as input Tensor of indexes and output an array of correspunding words
        
        input : Tensor of shape [ bptt, bsz]
        outptu : list of word strings of size [bptt, bsz]
        """
        bptt, bsz = indexes.shape
        
        output = [  [ 0  for i in range(bsz)] for j in range(bptt)]
        
        for timeStep in range(bptt):
            for batch in range(bsz):
                output[timeStep][batch] = TEXT.vocab.itos[indexes[timeStep, batch].cpu().long().item()]

              
        
        return(output)
        
        
    def from_input_to_before_last_layer(self, input, hid):
        """
        Take the input word indexes and process them until the penultimate layer 
        
        input : [timeStep, batchSize]
        output : [timeStep, batchSize, nhid]
        """
        
        emb = self.lut(input)
        hids, hid = self.rnn(emb, hid)    
        #did not insert dropout, to be checked...
        return(hids, hid)
        
        
    def forward(self, input, hid):
       """
       take as input the word indexes and outputs the log_probabilities of next word
       
       intput : [timeStep, batchSize]
       output : [timeStep, batchSize, vocab_Size]
       This output format is preferred due to the next loss layer, 
       instead of calling 2 times .view()
       """
       #safe check, in case of non complete batch (<self.args.bsz)
       local_time_length, local_batch_size = input.shape
       
       
       hids, hid = self.from_input_to_before_last_layer(input, hid)
       #print('CHECKUP')
       #print('huds', hids.shape)
       #print('hid', hid[0].shape)
       #print('hid', hid[1].shape)
        
       # Detach hiddens to truncate the computational graph for BPTT.
       # Recall that hid = (h,c).
       if self.args.loss == 'adaptive' :           
           log_prob=self.lastLayer.log_prob(hids.view(-1, self.nhid))
           #log_prob =self.lastLayer.log_prob(hids)
           
           return(log_prob.view(local_time_length, local_batch_size, self.vsize), hid)
           
       elif self.args.loss == 'regularCrossEntropy':
           logits=self.lastLayer(hids.view(-1,self.nhid))
           log_prob=self.get_logProb(logits)
           
           return (log_prob.view(local_time_length, local_batch_size, self.vsize), hid)
       
        
        
    def sample_from_log_prob(self, log_prob, mode='argmax'):
        """
        sample from the tensor of log_prob
        
        input : [bptt, bsz, vocab_size]
        output : [bptt, bsz]
        """
        
        
        prob=log_prob.exp()
        
        if mode == 'sample':
            #option 1 : sample
            bptt, bsz = log_prob.shape
            output=torch.zeros(bptt, bsz)
            for time_step in range(bptt):
                for batch in range(bsz):
                    output[time_step, batch]=torch.multinomial(prob[time_step,batch,:],1)
                    
        elif mode == 'argmax':
            #option 2 : argmax
            output=prob.argmax(dim=2)
            
        else:
            print('sampling mode unknown')
            
        return(output.long())
        
        
        
        
    
    def next_N_words(self, word, hid_, TEXT, length_to_predict):
        output=[]
        self.eval()
        for i in range(length_to_predict):
            wordIndex=TEXT.vocab.stoi[word]
            input_tensor=Variable(torch.LongTensor([[wordIndex]]))
            
            if torch.cuda.is_available():
                input_tensor=input_tensor.cuda(self.args.devid)
            
            emb = self.lut(input_tensor)
            
            hids, hid_ = self.rnn(emb, hid_)
            
            next_logits=self.proj((hids)).view(-1)
            next_distr=torch.nn.Softmax(dim=0)(torch.mul(next_logits, 1/self.temperature))
            next_index=torch.multinomial(next_distr,1)
            
            next_word=TEXT.vocab.itos[next_index.cpu().item()]
            output+=[next_word]
            word=next_word
        return(output)
    
    


    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        for v in h:
            v.detach_()
        return(h)
    

    def train_epoch(self, iter, val_iter,  gen_iter, loss, optimizer, viz, win, TEXT, args, infoToPlot=None):
        """
        Training loop
        """
        
        self.train()
        self.trainingBatches=0

        train_loss = 0
        nwords = 0

        hid = None
       
        iterable=tqdm(iter)
        
        for batch in iterable:
            self.trainingBatches+=1
            
            if hid is not None:
                #print(len(hid))
                hid=self.repackage_hidden(hid)
                #hid[0].detach_()
                #hid[1].detach_()
                
            optimizer.zero_grad()
            x = batch.text
            y = batch.target
            
            #print('x, ',x.shape)
            #print('y, ',y.shape)
            
            if torch.cuda.is_available():
                x=x.cuda(self.args.devid)
                y=y.cuda(self.args.devid)
                
           
            log_prob, hid = self(x, hid if hid is not None else None)
            
            #print('hid', hid[0].shape)
            #print('hid', hid[1].shape)
            #print('log_prob', log_prob.shape)
            #print('vocab ', self.vsize)
            
            #prediction = log_prob.view(-1, self.vsize)
            
            #print('prediction', prediction.shape)            
            #print('sum ', prediction.sum(dim=1).shape)
            #print('sum ', prediction.sum(dim=1)[:10])
            
            bloss = loss(log_prob.view(-1, self.vsize), y.contiguous().view(-1))

            #bloss = loss(log_prob.contiguous().view(-1, self.vsize), y.contiguous().view(-1))
            
            
            
            bloss.backward()
            train_loss += bloss.item()
            # bytetensor.sum overflows, so cast to int
            nwords += y.ne(self.padidx).int().sum().item()
            if self.args.clip > 0:
                clip_grad_norm_(self.parameters(), self.args.clip)

            optimizer.step()
            
            
            iterable.set_description("ppl %.3f" % np.exp(train_loss/nwords) )

            

            if self.batch_id % self.args.Nplot == 0:
                if not infoToPlot is None:
                    infoToPlot['trainPerp']+=[np.exp(train_loss/nwords)]
                    infoToPlot['id_batch_train']+=[self.batch_id]
                
                sampled_sentences=self.generate_predictions(gen_iter, TEXT)
                
                #print(sampled_sentences)
                infoToPlot['generated']=sampled_sentences
                #print(infoToPlot['generated'])
                
                estimate=self.sample_from_log_prob(log_prob) 
                                
                matching_train = y==estimate
                matching_train=matching_train.sum().cpu().data.tolist()           
                infoToPlot['matching_train']+=[matching_train/(y.shape[0] * y.shape[1])]
                
                updateDataStorage(infoToPlot, args.directoryData)                

                
                
                if self.args.vis:
                    win = visdom_plot(viz, win, self.args.directoryData)
                
                self.train()
            
                           
            
            if self.batch_id % self.args.Nsave == 0:
                
                valid_loss, valid_nwords = self.validate(val_iter, loss, viz, win, infoToPlot, max_number_of_iterations=np.inf)
                
                current_ppl=np.exp(valid_loss/valid_nwords)
                if current_ppl < self.best_ppl:
                    print('old best ppl : {} . New best ppl : {}'.format(self.best_ppl, current_ppl))
                    print('save model...')
                    self.save_model(args, bestPPL=current_ppl, mode='valid')
                    self.best_ppl=current_ppl
                    
                

            self.batch_id+=1
        self.trainingEpochs+=1
        
        return train_loss, nwords

    def validate(self, iter, loss, viz=None, win=None, infoToPlot=None, max_number_of_iterations=np.inf):
        """
        validation loop
        """
        
        self.eval()

        valid_loss = 0
        valid_nwords = 0

        hid = None
        
        iterable=tqdm(iter)        
        
        for counter,batch in enumerate(iterable):
            if counter>max_number_of_iterations:
                break
            else:               
                
                if hid is not None:
                    hid=self.repackage_hidden(hid)
                            
                x = batch.text
                y = batch.target
                
                if torch.cuda.is_available():
                    x=x.cuda(self.args.devid)
                    y=y.cuda(self.args.devid)
                    
                    
            
            log_prob, hid = self(x, hid if hid is not None else None)
            
           
            valid_loss += loss(log_prob.view(-1, self.vsize), y.contiguous().view(-1)).item()
                    
                     
            valid_nwords += y.ne(self.padidx).int().sum().item()
                
            iterable.set_description("ppl %.3f" % np.exp(valid_loss/valid_nwords))
                
                        
            
        if not infoToPlot is None:
            infoToPlot['validPerp']+=[np.exp(valid_loss/valid_nwords)]
            infoToPlot['id_batch_valid']+=[self.batch_id]
            
            
            #estimate=self.vec2idx(out)
            estimate=self.sample_from_log_prob(log_prob) 
            
            matching_valid = y==estimate
            matching_valid=matching_valid.sum().cpu().data.tolist()
            infoToPlot['matching_valid']+=[matching_valid/(y.shape[0] * y.shape[1])]
            
            
            
            if self.args.vis:
                updateDataStorage(infoToPlot, self.args.directoryData)
                win = visdom_plot(viz, win, self.args.directoryData, valid=True)
            
        self.train()
        return valid_loss, valid_nwords

    def generate_predictions(self, iterator, TEXT, outputDirectory=None, epoch=None, saveOutputs=False):
        """
        in-loop function that generates predictions using trained model        
        """
        
        
        if outputDirectory==None:
            outputDirectory=self.args.directoryData
        if epoch==None:
            epoch=self.trainingEpochs
        
            
        self.eval()
        
        batch_number=self.args.gen_bsz
        
        outputs=[dict({'input_warmup':'','output_warmup':'','input_sentence':'', 'output_sentence':''}) for _ in range(batch_number)]
        
        sample=next(iterator)
        
        input_warmup_idx=sample.text[:self.args.gen_warmup,:]
        input_sentence_idx = sample.text[self.args.gen_warmup:,:]
        
        
        if torch.cuda.is_available:
            input_warmup_idx=input_warmup_idx.cuda(self.args.devid)
            input_sentence_idx=input_sentence_idx.cuda(self.args.devid)
            
        
        
        #we will give the 20 first words of a sentence, and predict the 80 next characters
        input_warmup_words=self.idx2word(input_warmup_idx, TEXT)
        input_sentence_words=self.idx2word(input_sentence_idx, TEXT)
        
           
        if saveOutputs:
            with open(os.path.join(self.args.directoryData,self.__class__.__name__ + "_preds.txt"), "a") as f:
                f.write('*' * 20)
                f.write('\n \n NEW : EPOCH {} \n \n '.format(epoch))
                f.write('*' * 20)
                f.write('\n')
            
        #first we run the model on the first 20 words, in order to give context to the hidden state
        log_prob, hidden = self(input_warmup_idx, None)
        
        #next_distr=torch.nn.Softmax(dim=-1)(torch.mul(output_sentence.exp(), 1/self.temperature))
        output_warmup_idx = self.sample_from_log_prob(log_prob)
        
        #now we run the model in 'free wheel' using the generated predictions as input
        number_of_predictions = self.args.gen_bptt - self.args.gen_warmup
        
        output_sentence_idx=torch.ones(number_of_predictions, self.args.gen_bsz)
        
        
        start_word=input_sentence_idx[0,:].view(-1, self.args.gen_bsz)
        
        for timeStep in range(number_of_predictions):
            local_log_prob, hidden= self(start_word, hidden)
            start_word = self.sample_from_log_prob(local_log_prob)
            
            local_prediction=start_word.view(self.args.gen_bsz)
            output_sentence_idx[timeStep, :] = local_prediction
            
        #convert to words
        output_warmup_words=self.idx2word(output_warmup_idx, TEXT)
        output_sentence_words=self.idx2word(output_sentence_idx, TEXT)
            
    
        #print(output_warmup_words)
        #print(output_sentence_words)
            
        
        for batch in range(self.args.gen_bsz):
            for timeStep in range(self.args.gen_warmup):
                outputs[batch]['input_warmup']+=input_warmup_words[timeStep][batch] + ' ' 
                outputs[batch]['output_warmup']+=output_warmup_words[timeStep][batch] + ' ' 
            for timeStep in range(number_of_predictions):  
                outputs[batch]['input_sentence']+=input_sentence_words[timeStep][batch] + ' ' 
                outputs[batch]['output_sentence']+=output_sentence_words[timeStep][batch] + ' ' 

           
        if saveOutputs:
            with open(os.path.join(self.args.directoryData,self.__class__.__name__ + "_preds.txt"), "a") as f:
    
                f.write('input warmup : \n')
                f.write( outputs[0]['input_warmup'])
                f.write('\n \n')
                
                f.write('output warmup : \n')
                f.write( outputs[0]['output_warmup'])
                f.write('\n \n')
                
                f.write('input sentence : \n')
                f.write( outputs[0]['input_sentence'])
                f.write('\n \n')
                
                f.write('output sentence : \n')
                f.write( outputs[0]['output_sentence'])
                f.write('\n \n')
        
 
        return(outputs)
            
        
        
                
    def save_model(self, args, bestPPL, mode='train'):
        """
        Save the current model, working on a CPU
        """
               
        
        torch.save(self.cpu().state_dict(), os.path.join(args.directoryCkpt,self.args.model +'_Best{}_'.format(mode)+ "_ppl_" + '%.3f'%(bestPPL) +".pth"))

        
        if torch.cuda.is_available():
            self.cuda(self.args.devid)

