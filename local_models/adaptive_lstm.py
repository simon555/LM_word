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
from time import time
import pickle
import adaptive

#from local_models.log_uniform.log_uniform import LogUniformSampler

random.seed(1111)
torch.manual_seed(1111)


class LstmLm(nn.Module):
    def __init__(self, args, vocab, padidx):
        super(LstmLm, self).__init__()

        self.args=args
        self.nhid=args.nhid
        self.nlayers=args.nlayers
        self.dropout=args.dropout_LSTM
        self.tieweights=args.tieweights
        self.vsize = len(vocab.itos)
        self.padidx=padidx
        
        #number of batches of training seen
        self.batch_id=0
        
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
        self.decoder=adaptive.AdaptiveLogSoftmaxWithLoss(self.nhid, 
                                                         self.vsize,
                                                         [int(self.vsize*0.2),int(self.vsize*0.5),int(self.vsize*0.7), int(self.vsize*0.9)])
                                                        # [int(self.vsize*0.1), int(self.vsize*0.3), int(self.vsize*0.7)])
            
        
        
        self.best_ppl=np.inf
        
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
        return (self.drop(hids), hid)
    
    def next_N_words(self, word, hid_, TEXT, length_to_predict):
        output=[]
        self.eval()
        for i in range(length_to_predict):
            wordIndex=TEXT.vocab.stoi[word]
            input_tensor=Variable(torch.LongTensor([[wordIndex]]))
            
            if torch.cuda.is_available():
                input_tensor=input_tensor.cuda()
            
            emb = self.lut(input_tensor)
            
            hids, hid_ = self.rnn(emb, hid_)
            
            #next_logits=self.proj((hids)).view(-1)
            #next_distr=torch.nn.Softmax(dim=0)(torch.mul(next_logits, 1/self.temperature))
            #next_index=torch.multinomial(next_distr,1)
            
            next_distribution = torch.exp(self.decoder.log_prob(hids[-1,:,:]))
            next_index=torch.multinomial(next_distribution, 1)
            
            next_word=TEXT.vocab.itos[next_index.cpu().item()]
            output+=[next_word]
            word=next_word
        return(output)

    def repackage_hidden(self,h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        for v in h:
            v.detach_()
    

    def train_epoch(self, iter, val_iter,  loss, optimizer, viz, win, TEXT, args, infoToPlot=None):
        self.train()
        self.trainingBatches=0

        train_loss = 0
        nwords = 0

        hid = None
       
        iterable=tqdm(iter)
        train_mean_time=0
        
        for local_batch_number, batch in enumerate(iterable):
            start=time()
            self.trainingBatches+=1
            
            if hid is not None:
                #print(len(hid))
                self.repackage_hidden(hid)
                #hid[0].detach_()
                #hid[1].detach_()
                
            optimizer.zero_grad()
            x = batch.text
            y = batch.target
            
            if torch.cuda.is_available():
                x=x.cuda()
                y=y.cuda()
                
           
            out, hid = self(x, hid if hid is not None else None)
            #print('hid', hid[0].shape)
            #print('hid', hid[1].shape)
            
            #bloss = loss(out.view(-1, self.vsize), y.contiguous().view(-1))
                        
            bloss = self.decoder(out.contiguous().view(-1,self.nhid), y.contiguous().view(-1)).loss

            bloss.backward()
            train_loss += bloss.item()
            # bytetensor.sum overflows, so cast to int
            nwords += y.ne(self.padidx).int().sum().item()
            if self.args.clip > 0:
                clip_grad_norm_(self.parameters(), self.args.clip)

            optimizer.step()
            
            end=time()
            
            batch_time=end-start
            
            train_mean_time += batch_time
            
            
            iterable.set_description("ppl %.3f" % np.exp(train_loss/nwords) )
            
            

            if self.batch_id % self.args.Nplot == 0:
                if not infoToPlot is None:
                    infoToPlot['trainPerp']+=[np.exp(train_loss/nwords)]
                    infoToPlot['id_batch_train']+=[self.batch_id]
                    infoToPlot['time_train']+=[train_mean_time/(local_batch_number+1)]
                
                sampled_sentences=self.generate_predictions(TEXT)
                #print(sampled_sentences)
                infoToPlot['generated']=sampled_sentences
                #print(infoToPlot['generated'])
                if self.args.vis:
                    win = visdom_plot(viz, win, infoToPlot)
                
                self.train()
            
            if self.batch_id % self.args.NinfoSave == 0:
                with open(os.path.join(args.directoryData,'data.pkl'), 'wb') as f:
                    pickle.dump(infoToPlot, f, pickle.HIGHEST_PROTOCOL)
            
            if self.batch_id % self.args.Nsave == 0:
                
                valid_loss, valid_nwords = self.validate(val_iter, loss, viz, win, infoToPlot, max_number_of_iterations=1000)
                
                current_ppl=np.exp(valid_loss/valid_nwords)
                if current_ppl < self.best_ppl:
                    print('old best ppl : {} . New best ppl : {}'.format(self.best_ppl, current_ppl))
                    print('save model...')
                    self.save_model(args, mode='valid')
                    self.best_ppl=current_ppl
                    
                

            self.batch_id+=1
        self.trainingEpochs+=1
        
        return train_loss, nwords

    def validate(self, iter, loss, viz=None, win=None, infoToPlot=None, max_number_of_iterations=np.inf):
        self.eval()

        valid_loss = 0
        nwords = 0

        hid = None
        
        iterable=tqdm(iter)        
        mean_time=0
        for counter,batch in enumerate(iterable):
            start=time()
            if counter>max_number_of_iterations:
                break
            else:               
                
                if hid is not None:
                    hid[0].detach_()
                    hid[1].detach_()
                            
                x = batch.text
                y = batch.target
                
                if torch.cuda.is_available():
                    x=x.cuda()
                    y=y.cuda()
                    
                    
                out, hid = self(x, hid if hid is not None else None)
                
                
                valid_loss += self.decoder(out.contiguous().view(-1,self.nhid), y.contiguous().view(-1)).loss.item()
                
                
                nwords += y.ne(self.padidx).int().sum().item()
                
                end=time()
                batch_time=end-start
                mean_time += batch_time
                
                
                iterable.set_description("ppl %.3f" % np.exp(valid_loss/nwords))
                
                
            
        if not infoToPlot is None:
            infoToPlot['validPerp']+=[np.exp(valid_loss/nwords)]
            infoToPlot['id_batch_valid']+=[self.batch_id]
            infoToPlot['time_valid']+=[mean_time/(counter+1)]
            if self.args.vis:
                win = visdom_plot(viz, win, infoToPlot, valid=True)
            
        self.train()
        return valid_loss, nwords

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
            if os.name=='nt':
                data = torchtext.datasets.LanguageModelingDataset(
                path=os.path.join(os.getcwd(),"data", "splitted", "smallData","gen.txt"),
                text_field=TEXT)
            else:
                data = torchtext.datasets.LanguageModelingDataset(
                path=os.path.join('/mnt','raid1','text','big_files','splitted','springer_cui_tokenized','test.txt'),
                text_field=TEXT)
            
            
            data_iter = torchtext.data.BPTTIterator(data, batch_number, bptt_len=100, train=False)
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
        #print(expected_output_words)
        
       
        if saveOutputs:
            with open(os.path.join(self.args.directoryData,self.__class__.__name__ + "_preds.txt"), "a") as f:
                f.write('*' * 20)
                f.write('\n \n NEW : EPOCH {} \n \n '.format(epoch))
                f.write('*' * 20)
                f.write('\n')
            
        #first we run the model on the first 20 words, in order to give context to the hidden state
        output_sentence, hidden = self(input_sentence, None)
            
        for batch_index in range(batch_number):          
            
            #then we run the model using the  computed hidden states
            #len_sent=len(expected_output_words[batch_index])
            #max_index=min(len_sent, 20)
            max_index=len(expected_output_words[batch_index])
            #print('max index ', max_index)
            index=min(max_index-1, 20)
            
            input_word=expected_output_words[batch_index][index]
            
            #we select the correct hidden state for the batch
            #because next_N_words take a single word as input
            local_h=hidden[0][:,batch_index:batch_index+1,:]
            local_c=hidden[1][:,batch_index:batch_index+1,:]
            local_hidden=(local_h.contiguous(), local_c.contiguous())
            
            output=self.next_N_words(input_word, local_hidden, TEXT, 80)

            outputs[batch_index]['input_sentence']=''.join([x+' ' for x in input_words[batch_index]])
            outputs[batch_index]['expected_sentence']=''.join([x+' ' for x in expected_output_words[batch_index]])
            outputs[batch_index]['output_sentence']=''.join([x+' ' for x in output])
            
            if saveOutputs:
                with open(os.path.join(self.args.directoryData,self.__class__.__name__ + "_preds.txt"), "a") as f:
    
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
            
                
    def save_model(self, args, mode='train'):
        copy_of_gen_iterator=self.iterator
        
        self.generationIteratorBuilt=False        
        self.iterator=None
        
        torch.save(self.cpu().state_dict(), os.path.join(args.directoryCkpt,self.args.model +'_Best{}_'.format(mode)+ ".pth"))
        


        self.iterator=copy_of_gen_iterator
        self.generationIteratorBuilt=True
        
        if torch.cuda.is_available():
            self.cuda()



