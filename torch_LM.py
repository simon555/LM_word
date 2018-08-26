# -*- coding: utf-8 -*-


# =============================================================================
# summarizing the arguments for this experiment
# =============================================================================
print('parse arguments...')
from arguments import get_args
args = get_args()



# =============================================================================
# specifying the CUDA devide available on the server
# =============================================================================
import os

#check if you work  on a local windows machine to debug, or on a Linux server
if os.name != 'nt':
    #work on linux server
    command_to_run='CUDA_VISIBLE_DEVICE={}'.format(args.devid)
    check=os.system(command_to_run)
    print('command run : ', command_to_run, '...', check==0)
    




print('loading dependencies...')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable as V
import os
import torchtext
import time
import random
import sys
import math
from local_models import lstm, adaptive_lstm
from losses import regularCrossEntropy
import lazyContiguousDataset
import dill as pickle
from visualisation import visdom_plot
from shutil import copyfile
import infoToTrack
import visualisation
import itertools
from infoToTrack import updateDataStorage

random.seed(1111)
torch.manual_seed(1111)


# =============================================================================
# load the data
# =============================================================================
print('load the data... ')

# first we check if a vocab has been already built
#in torchtext, vocabs are stored within the class `Field`
print('setup the vocab')
#try to load a premade vocab
if os.name=='nt':
    root=os.path.join(os.getcwd(),'data','splitted')
else:
    root=os.path.join('/mnt','raid1','text','big_files','splitted')

#check max_size of vocabulary
if args.vocab_size!=-1:
    size_of_voc=args.vocab_size
else:
    size_of_voc='full'
    
#grainLevel can be word or char
vocab_folder=os.path.join(root, args.dataset, 'vocab',args.grainLevel)

path_to_vocab=os.path.join(vocab_folder, 'vocab_{}_{}.pickle'.format(size_of_voc,args.grainLevel))


try:
    print('trying to load premade vocab...')
    TEXT=pickle.load(open(path_to_vocab,'rb'))
    print('premade vocab loaded : ', path_to_vocab)
    
except:
    print('premade vocab not found, build a new one...')
    TEXT = torchtext.data.Field()    
    
    
    
# =============================================================================
# BUILD DATASET    
# =============================================================================
    
print('build dataset...')
if os.name=='nt':
    pathToData=os.path.join(os.getcwd(),'data','splitted','smallData')
else:
    pathToData=os.path.join('/mnt','raid1','text','big_files','splitted','springer_cui_tokenized')
        
    

if args.lazyLoading is False:
    print('you should use lazy loading if you see that your dataset does not fit in memory')
    train, valid, test = torchtext.datasets.LanguageModelingDataset.splits(
path=pathToData, train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)

else:
    print('using lazy loading')
    train, valid, test = lazyContiguousDataset.LanguageModelingDataset.splits(
            path=pathToData, train="train.txt", validation="valid.txt", test="test.txt", args=args, text_field=TEXT)
   



#we then create a different generator to generate ramdom sentences 
#that continue sentences from the test dataset
# it has different bsz and bptt parameters
if os.name=='nt':
    data = torchtext.datasets.LanguageModelingDataset(
    path=os.path.join(os.getcwd(),"data", "splitted", "smallData","test.txt"),
    text_field=TEXT)
else:
    data = torchtext.datasets.LanguageModelingDataset(
    path=os.path.join('/mnt','raid1','text','big_files','splitted','springer_cui_tokenized','test.txt'),
    text_field=TEXT)
    
data_iter = torchtext.data.BPTTIterator(data, batch_size=args.gen_bsz, bptt_len=args.gen_bptt, train=False)
gen_iterator=itertools.cycle(iter(data_iter))


#check if the vocab of the text field is built or no 
#depending on loading a premade vocab or building a new one
try:
    #mere check to see if TEXT.vocab is initialized
    print('vocab of size ', len(TEXT.vocab.itos))

except:
    print('build vocab...')
    
    if args.lazyLoading is False:
        #without lazy loading, the vocab is built with the full dataset available in memory
        TEXT.build_vocab(train, max_size=args.vocab_size if args.vocab_size is not -1 else None )
    else:
        #we implemented a lazy vodab builder, available via the LazyDataset class
        #the field whose vocab is built is the same one that was used to create the train dataset
        train.build_vocab(max_size=args.vocab_size if args.vocab_size is not -1 else None )

    print('vocab built, we save it for a later use')
    if not os.path.exists(vocab_folder):
        os.makedirs(vocab_folder)
        
    with open(path_to_vocab, 'wb') as handle:
            pickle.dump(TEXT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('vocab saved!')
    
    
print('We will use a vocab of size ', len(TEXT.vocab.itos))
args.fullSize=len(TEXT.vocab.itos)


#check the indexes of special tokens
unkidx = TEXT.vocab.stoi["<unk>"]
padidx = TEXT.vocab.stoi["<pad>"]


#build data iterators
print('build iterators...')
if args.lazyLoading is False:
    print('you should use lazy loading if you have memory issues')
    train_iter, valid_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, valid, test), batch_size=args.bsz, bptt_len=args.bptt, repeat=False)

else:
    train_iter, valid_iter, test_iter = lazyContiguousDataset.lazy_BPTTIterator.splits(
    (train, valid, test), batch_size=args.bsz, bptt_len=args.bptt, repeat=False, shuffle=True, device=0)






# =============================================================================
# Build the output directories for this experiment
# =============================================================================
print('build output directories for this experiment...')

#In this framework, we save the experiments using a precise denomination

#ID of the experiment, if similar ones have been run
Nexperience=1

directoryOut = os.path.join(os.getcwd(), 'results', '{}_{}_Exp{}'.format(args.model,args.dataset,Nexperience))
   

#if similar experiments have already been run (model/dataset/...) 
#we create a new folder with a unique ID 
if not os.path.exists(directoryOut):
    print('new directory : ',directoryOut)        
else:
    while(os.path.exists(directoryOut)):
        Nexperience+=1
        directoryOut = os.path.join(os.getcwd(), 'results', '{}_{}_Exp{}'.format(args.model,args.dataset,Nexperience))
    print('new directory : ',directoryOut)
        
#build subdirectories
directoryCkpt=os.path.join(directoryOut,'checkpoint')
directoryData=os.path.join(directoryOut,'data')
directoryScripts=os.path.join(directoryOut, 'scrpits')
        
os.makedirs(directoryOut) 
os.makedirs(directoryData)
os.makedirs(directoryCkpt)
os.makedirs(directoryScripts)

#add these info the  args file
d=vars(args)
d['directoryOut']=directoryOut
d['directoryData']=directoryData
d['directoryCkpt']=directoryCkpt
d['directoryScripts']=directoryScripts
d['Nexperience']=Nexperience



# =============================================================================
# save the scripts to rerun this experiment
# =============================================================================
print('save the informations of this experiment...')

#save the model scripts
src = os.path.join(os.getcwd(),'local_models','{}.py'.format(args.model))
trg = os.path.join(directoryScripts, '{}.py'.format(args.model))     
copyfile(src,trg)
    

#save the arguments scripts
src = os.path.join(os.getcwd(),'arguments.py')
trg = os.path.join(directoryScripts, 'arguments.py')   
copyfile(src,trg)


#save a descriptor of the experiment    
descriptor=''
for i in vars(args):
    line_new = '{:>12}  {:>12} \n'.format(i, getattr(args,i))
    descriptor+=line_new
print(descriptor)

with open(os.path.join(directoryScripts, 'descriptor.txt'), 'w') as outstream:
    outstream.write("experience started on : {} at {}  \n".format(time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))
    outstream.write(descriptor)
    

     
# =============================================================================
# Define the info to track along the experiment
# =============================================================================
infoToPlot=infoToTrack.getInfo()
with open(os.path.join(directoryData,'data.pkl'), 'wb') as f:
    pickle.dump(infoToPlot, f, pickle.HIGHEST_PROTOCOL)
# =============================================================================
# Define the Vizdom visualizer
# =============================================================================
if args.vis:
    
    viz, win = visualisation.setupViz(args, descriptor)
    






if __name__ == "__main__":
    
# =============================================================================
#     Build the model
# =============================================================================
    print('build model...')
    if args.model=='lstm':
        model = lstm.LstmLm(args,TEXT.vocab, padidx)
    elif args.model=='adaptive_lstm':
        model = adaptive_lstm.LstmLm(args,TEXT.vocab, padidx)
    else:
        print('model not implemented : ',args.model)
    print(model)
    
    
# =============================================================================
#    Select the loss to use
# =============================================================================
    loss=regularCrossEntropy.Loss(vsize=model.vsize,padidx=padidx, unkidx=unkidx)

#    if args.loss=='regularCrossEntropy':
#        loss=regularCrossEntropy.Loss(vsize=model.vsize,padidx=padidx, unkidx=unkidx, size_average=False)
#    elif args.loss=='adaptive':
#        #TO IMPLEMENT !!!!
#        loss=adaptiveSoftMax()
#    else:
#        print('loss not implemented : ', args.loss)
    
        
    
    
    if torch.cuda.is_available():
        print('with cuda!')
        loss=loss.cuda()
         #criterion=criterion.cuda()


# =============================================================================
#   Define optimizer
# =============================================================================
    print('define optimizer...')
    params_model = [p for p in model.parameters() if p.requires_grad]
    params_loss = [p for p in loss.parameters() if p.requires_grad]
    params = params_model + params_loss
    
    
    if args.optim == "Adam":
        optimizer = optim.Adam(
            params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))
    elif args.optim == "SGD":
        optimizer = optim.SGD(
            params, lr = args.lr, weight_decay = args.wd,
            nesterov = not args.nonag, momentum = args.mom, dampening = args.dm)
    
    #schedule for the learning rate
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=args.lrd, threshold=1e-3)


# =============================================================================
#   Training loop
# =============================================================================
    print('start the training...')
    for epoch in range(args.epochs):
        print("Epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss, train_words = model.train_epoch(
            iter=train_iter, val_iter = valid_iter, gen_iter=gen_iterator, loss=loss, optimizer=optimizer,infoToPlot=infoToPlot, viz=viz, win=win, TEXT=TEXT, args=args)
        
        valid_loss, valid_words = model.validate(valid_iter, loss, infoToPlot=infoToPlot, viz=viz, win=win)
        
        if args.scheduleLearningRate==True:
            schedule.step(valid_loss)
        
            
        #show perplexity on train and test at the end of each epoch
        print("Train: {}, Valid: {}".format(
            math.exp(train_loss / train_words), math.exp(valid_loss / valid_words)))
        
        #save info of the epoch
        print('save info...')
        updateDataStorage(infoToPlot, directoryData)
        
        #generate samples to display via VIZDOM
        print('generate sameple sentences...')    
        outputs=model.generate_predictions(gen_iterator, TEXT, saveOutputs= True)
        infoToPlot['generated']=outputs
        win = visdom_plot(viz, win, directoryData ,valid=True)

    #test and save model
    print('run on the test set...')
    test_loss, test_words = model.validate(test_iter, loss)
    print("Test: {}".format(math.exp(test_loss / test_words)))
    print('generate predictions with the trained model...')
    model.generate_predictions(TEXT)
    print('save model...')
    model.save_model(args)
    
    
    #indicate the end date of the experiment
    with open(os.path.join(directoryScripts, 'descriptor.txt'), 'w') as outstream:
        outstream.write("experience finished on : {} at {}  \n".format(time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))
    
    
    print('done')