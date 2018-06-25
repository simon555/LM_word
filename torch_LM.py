# -*- coding: utf-8 -*-

print('loading dependencies...')
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
from torch.autograd import Variable as V
from torch.nn import Parameter
from torch.nn.utils import clip_grad_norm
import os
import torchtext
import time
from tqdm import tqdm
import random
import sys
import math
from arguments import get_args
from local_models import lstm

import dill as pickle
# =============================================================================
# summarizing the arguments for this experiment
# =============================================================================
print('parse arguments...')

args = get_args()


random.seed(1111)
torch.manual_seed(1111)
if args.devid >= 0:
    torch.cuda.manual_seed_all(1111)
    torch.backends.cudnn.enabled = False
    print("Cudnn is enabled: {}".format(torch.backends.cudnn.enabled))


# =============================================================================
# load the data
# =============================================================================
print('load the data...')


#try to load a premade vocab
if os.name=='nt':
    root=os.path.join(os.getcwd(),'data','splitted')
else:
    root=os.path.join('/mnt','raid1','text','big_files','splitted')


if args.vocab_size is not False:
    size_of_voc=args.vocab_size
else:
    size_of_voc='full'
    
    
vocab_folder=os.path.join(root, args.dataset, 'vocab')
path_to_vocab=os.path.join(vocab_folder, 'vocab_{}_words.pickle'.format(size_of_voc))


try:
    print('trying to load premade vocab...')
    TEXT=pickle.load(open(path_to_vocab,'rb'))
    print('premade vocab loaded : ', path_to_vocab)
    
except:
    print('premade vocab not found, build a new one...')
    #load raw data
    print('load data...')
    TEXT = torchtext.data.Field()    
    if os.name=='nt':
        pathToData=os.path.join(os.getcwd(),'data','splitted','smallData')
    else:
        pathToData=os.path.join('/mnt','raid1','text','big_files','splitted','springer_cui_tokenized')
        

    train, valid, test = torchtext.datasets.LanguageModelingDataset.splits(
    path=pathToData,
    train="train.txt", validation="valid.txt", test="valid.txt", text_field=TEXT)
    #path="data/",
    #train="train.txt", validation="valid.txt", test="test.txt", text_field=TEXT)
    print('build vocab...')
    TEXT.build_vocab(train, max_size=args.vocab_size if args.vocab_size is not False else None )
    
    print('vocab built, we save it for a later use')
    if not os.path.exists(vocab_folder):
        os.makedirs(vocab_folder)
        
    with open(path_to_vocab, 'wb') as handle:
            pickle.dump(TEXT, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print('vocab saved!')
    
padidx = TEXT.vocab.stoi["<pad>"]



#build data iterators
print('build iterators...')
train_iter, valid_iter, test_iter = torchtext.data.BPTTIterator.splits(
    (train, valid, test), batch_size=args.bsz, device=0, bptt_len=args.bptt, repeat=False)



# =============================================================================
# Build the output directories for this experiment
# =============================================================================
print('build output directories for this experiment...')

Nexperience=1

directoryOut = os.path.join(os.getcwd(), 'results', '{}_{}_Exp{}'.format(args.model,args.dataset,Nexperience))
    
if not os.path.exists(directoryOut):
    print('new directory : ',directoryOut)        
else:
    while(os.path.exists(directoryOut)):
        print('directory already exists : ',directoryOut)
        Nexperience+=1
        directoryOut = os.path.join(os.getcwd(), 'results', '{}_{}_Exp{}'.format(args.model,args.dataset,Nexperience))
    print('new directory : ',directoryOut)
        
directoryCkpt=os.path.join(directoryOut,'checkpoint')
directoryData=os.path.join(directoryOut,'data')
directoryScripts=os.path.join(directoryOut, 'scrpits')
        
os.makedirs(directoryOut) 
os.makedirs(directoryData)
os.makedirs(directoryCkpt)
os.makedirs(directoryScripts)

#add these info the the args file
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
if os.name=='nt':
    root='copy '
else:
    root='cp '

commandBash = root + os.path.join(os.getcwd(),'local_models','lstm.py')
commandBash += ' '
commandBash += os.path.join(directoryScripts, 'model.py')   
    
check=os.system(commandBash)
if check==1:
    print(commandBash)
    sys.exit("ERROR, model not copied")
    
#save the arguments scripts, as well as the descriptor of the experiment
commandBash = root + os.path.join(os.getcwd(),'arguments.py')
commandBash += ' '
commandBash += os.path.join(directoryScripts, 'arguments.py')   



    
check=os.system(commandBash)
if check==1:
    print(commandBash)
    sys.exit("ERROR, argument script not copied")
    
    
descriptor=''
for i in vars(args):
    #print(i, getattr(args,i))
    line_new = '{:>12}  {:>12} \n'.format(i, getattr(args,i))
    descriptor+=line_new
    #print(line_new)
print(descriptor)
#time.sleep(5)

with open(os.path.join(directoryScripts, 'descriptor.txt'), 'w') as outstream:
    outstream.write("experience done on : {} at {}  \n".format(time.strftime("%d/%m/%Y"),time.strftime("%H:%M:%S")))
    outstream.write(descriptor)
    

     
# =============================================================================
# Define the info to track along the experiment
# =============================================================================
infoToPlot={'trainPerp':[],
            'validPerp':[],
            'generated':None}


# =============================================================================
# Define the Vizdom visualizer
# =============================================================================
if args.vis:
    from visdom import Visdom
    print('using VISDOM')
    #specific to local experiments on windows laptop [Simon]
   
    viz = Visdom(server=args.serverVisdom,port=args.portVisdom,env='{}_{}_Exp{}/'.format(args.model,args.dataset,Nexperience))
    win={'trainPerp':None,
         'validPerp':None,
         'input_sentence':None,
         'expected_sentence':None,
         'output_sentence':None}
    #plot a summary of the experiment on visdom
    winForDescriptor=viz.text(descriptor)







if __name__ == "__main__":
    
    #instantiate model
    print('build model...')
    model = lstm.LstmLm(args,TEXT.vocab, padidx)
    
    print(model)
    
    if args.devid >= 0:
        model.cuda(args.devid)

    # We do not want to give the model credit for predicting padding symbols,
    # this can decrease ppl a few points.
    weight = torch.FloatTensor(model.vsize).fill_(1)
    weight[padidx] = 0
    if torch.cuda.is_available():
        weight = weight.cuda()
    loss = nn.CrossEntropyLoss(weight=V(weight), size_average=False)
    
    if torch.cuda.is_available():
        print('with cuda!')
        loss=loss.cuda()


    #define optimizer
    print('define optimizer...')
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "Adam":
        optimizer = optim.Adam(
            params, lr = args.lr, weight_decay = args.wd, betas=(args.b1, args.b2))
    elif args.optim == "SGD":
        optimizer = optim.SGD(
            params, lr = args.lr, weight_decay = args.wd,
            nesterov = not args.nonag, momentum = args.mom, dampening = args.dm)
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=args.lrd, threshold=1e-3)


    #training loop
    print('start the training...')
    for epoch in range(args.epochs):
        print("Epoch {}, lr {}".format(epoch, optimizer.param_groups[0]['lr']))
        train_loss, train_words = model.train_epoch(
            iter=train_iter, loss=loss, optimizer=optimizer,infoToPlot=infoToPlot, viz=viz, win=win, TEXT=TEXT)
        valid_loss, valid_words = model.validate(valid_iter, loss, infoToPlot=infoToPlot, viz=viz, win=win)
        schedule.step(valid_loss)
        print("Train: {}, Valid: {}".format(
            math.exp(train_loss / train_words), math.exp(valid_loss / valid_words)))
        
        print('save info...')
        with open(os.path.join(directoryData,'data.pkl'), 'wb') as f:
            pickle.dump(infoToPlot, f, pickle.HIGHEST_PROTOCOL)
        print('generate sameple sentences...')    
        outputs=model.generate_predictions(TEXT, saveOutputs=True)

    #test and save model
    print('run on the test set...')
    test_loss, test_words = model.validate(test_iter, loss)
    print("Test: {}".format(math.exp(test_loss / test_words)))
    print('generate predictions with the trained model...')
    model.generate_predictions(TEXT)
    print('save model...')
    model.generationIteratorBuilt=False
    model.iterator=None
    torch.save(model.cpu().state_dict(), os.path.join(directoryCkpt,model.__class__.__name__ + ".pth"))
    print('done')