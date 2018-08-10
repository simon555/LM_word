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
import time
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
        self.proj = nn.Linear(self.nhid, self.vsize)
        self.trainingEpochs=0 #number of epochs already trained
        self.trainingBatches=0
        self.temperature=1
        self.generationIteratorBuilt=False
        
        
        
        self.best_ppl=np.inf
        
        if self.tieweights:
            # See https://arxiv.org/abs/1608.05859
            # Seems to improve ppl by 13%.
            self.proj.weight = self.lut.weight
        
        if torch.cuda.is_available():
            self.cuda()
    
    
    def vec2idx(self, logits, mode='argmax'):        
        '''
        convert the output of forward into a sequence of word indexes
        input : [bptt, Nbatch, vsize]
        output : [bptt, Nbatch]
        '''       
        distribution = torch.nn.Softmax(dim=-1)(torch.mul(logits,1/self.temperature))
    
        
        if mode=='sample':
            output=[]        
            for timeStep in range(distribution.shape[0]):
                output.append(torch.multinomial(distribution[timeStep,:,:],1).view(-1).cpu().data.tolist())

        elif mode=='argmax':
            val, output = distribution.max(dim=-1)
        
        else:
            print('mode of sampling not found : ', mode)
        
        return(output)
        
        
        
        
    def forward(self, input, hid):
        emb = self.lut(input)
        hids, hid = self.rnn(emb, hid)
        # Detach hiddens to truncate the computational graph for BPTT.
        # Recall that hid = (h,c).
        return (self.proj(self.drop(hids)), hid)
    
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
    

    def train_epoch(self, iter, val_iter,  loss, optimizer, viz, win, TEXT, args, infoToPlot=None):
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
            
            if torch.cuda.is_available():
                x=x.cuda()
                y=y.cuda()
                
           
            out, hid = self(x, hid if hid is not None else None)
            #print('hid', hid[0].shape)
            #print('hid', hid[1].shape)
            bloss = loss(out.contiguous().view(-1, self.vsize), y.contiguous().view(-1))
            
            
            
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
                
                sampled_sentences=self.generate_predictions(TEXT)
                #print(sampled_sentences)
                infoToPlot['generated']=sampled_sentences
                #print(infoToPlot['generated'])
                
                estimate=self.vec2idx(out)            
                matching_train = y==estimate
                matching_train=matching_train.sum().cpu().data.tolist()           
                infoToPlot['matching_train']+=[matching_train/(y.shape[0] * y.shape[1])]
                
                
                
                if self.args.vis:
                    win = visdom_plot(viz, win, infoToPlot)
                
                self.train()
            
            if self.batch_id % 100 * self.args.Nplot == 0:
                with open(os.path.join(args.directoryData,'data.pkl'), 'wb') as f:
                    pickle.dump(infoToPlot, f, pickle.HIGHEST_PROTOCOL)
            
            if self.batch_id % self.args.Nsave == 0:
                
                valid_loss, valid_nwords = self.validate(val_iter, loss, viz, win, infoToPlot, max_number_of_iterations=np.inf)
                
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
                    x=x.cuda()
                    y=y.cuda()
                    
                    
                     
            out, hid = self(x, hid if hid is not None else None)
            valid_loss += loss(out.contiguous().view(-1, self.vsize), y.contiguous().view(-1)).item()
            valid_nwords += y.ne(self.padidx).int().sum().item()
                
            iterable.set_description("ppl %.3f" % np.exp(valid_loss/valid_nwords))
                
                        
            
        if not infoToPlot is None:
            infoToPlot['validPerp']+=[np.exp(valid_loss/valid_nwords)]
            infoToPlot['id_batch_valid']+=[self.batch_id]
            
            
            estimate=self.vec2idx(out)
            matching_valid = y==estimate
            matching_valid=matching_valid.sum().cpu().data.tolist()
            infoToPlot['matching_valid']+=[matching_valid/(y.shape[0] * y.shape[1])]
            
            
            
            if self.args.vis:
                win = visdom_plot(viz, win, infoToPlot, valid=True)
            
        self.train()
        return valid_loss, valid_nwords

    def generate_predictions(self, TEXT, outputDirectory=None, epoch=None, saveOutputs=False):
        if outputDirectory==None:
            outputDirectory=self.args.directoryData
        if epoch==None:
            epoch=self.trainingEpochs
        
            
        self.eval()
        batch_number=10
        outputs=[dict({'input_sentence':'','expected_sentence':'','output_sentence':'', 'TF_output':''}) for _ in range(batch_number)]
        #prepare the data to generate on
        
        if not self.generationIteratorBuilt:
            if os.name=='nt':
                data = torchtext.datasets.LanguageModelingDataset(
                path=os.path.join(os.getcwd(),"data", "splitted", "smallData","train.txt"),
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
        
        next_distr=torch.nn.Softmax(dim=-1)(torch.mul(output_sentence, 1/self.temperature))


        for batch_index in range(batch_number):          
            
            #then we run the model using the  computed hidden states
            #len_sent=len(expected_output_words[batch_index])
            #max_index=min(len_sent, 20)
            max_index=len(expected_output_words[batch_index])-1
            #print(max_index)      
            if max_index <20 :
                pass
            else:
                #print('max index ', max_index)
                index=20
                expected_output_words[batch_index] = expected_output_words[batch_index][index:]
                input_word=expected_output_words[batch_index][0]
                
                #we select the correct hidden state for the batch
                #because next_N_words take a single word as input
                local_h=hidden[0][:,batch_index:batch_index+1,:].contiguous()
                local_c=hidden[1][:,batch_index:batch_index+1,:].contiguous()
                local_hidden=(local_h, local_c)
                
                output=self.next_N_words(input_word, local_hidden, TEXT, 80)
                
                
                outputs[batch_index]['input_sentence']=''.join([x+' ' for x in input_words[batch_index]])
                outputs[batch_index]['expected_sentence']=''.join([x+' ' for x in expected_output_words[batch_index]])
                outputs[batch_index]['output_sentence']=''.join([x+' ' for x in output])
                
                next_index=torch.multinomial(next_distr[:,batch_index,:],1)
                outputs[batch_index]['TF_output']=''.join([TEXT.vocab.itos[x] + ' '  for x in next_index.view(-1).data.tolist()])

                
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
    
    


