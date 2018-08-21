# -*- coding: utf-8 -*-


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
matplotlib.rcParams.update({'font.size': 8})

from torchvision.utils import make_grid
from visdom import Visdom


def setupViz(args, descriptor):
    from visdom import Visdom

    print('using VISDOM')
    #specific to local experiments on windows laptop [Simon]
   
    viz = Visdom(server=args.serverVisdom,port=args.portVisdom,env='{}_{}_Exp{}/'.format(args.model,args.dataset,args.Nexperience))
    win={'trainPerp':None,
         'validPerp':None,
         'input_warmup':None,
         'output_warmup':None,
         'input_sentence':None,
         'output_sentence':None,
         'time_train':None,
         'time_valid':None,
         'matching_train':None,
         'matching_valid':None}
    
    #plot a summary of the experiment on visdom
    winForDescriptor=viz.text(descriptor)
    
    return(viz, win)

def load_from_results(path):
    '''
    open a viz server and plot the results obtained from the path of the saved info
    '''
    import pickle 
    
    infoToPlot=pickle.load(path)
    
    nameOfExp='' 
    
    splittedPath=path.split('/')
    i=0
    while len(nameOfExp) == 0 : 
        nameOfExp=splittedPath[-i]
        i+=1
    
    viz = Visdom(server='http://localhost',port=8097,env=nameOfExp)
    win={'trainPerp':None,
         'validPerp':None,
         'input_sentence':None,
         'expected_sentence':None,
         'output_sentence':None}
    
    win=plotTrainPPL(infoToPlot['trainPerp'], viz, win)
    
    win=plotValidPPL(infoToPlot['validPerp'], viz, win)
    
    win=plotSampledSentences(infoToPlot['generated'], viz, win)
    

def plotTrainPPL(dataTrain, indexes,  infoToPlot, viz, win):  
    NTrain=len(dataTrain)    
   

    dataTrain=np.array(dataTrain)
    indexes=np.array(indexes)
    localWin =viz.line(
       Y=dataTrain,
       X=indexes,
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='batch',
            ylabel='Train perplexity',
            title='evolution of the perplexity on training set',
            legend=['trainPerp'],
            ) ,win=win['trainPerp']
        )
        
    win['trainPerp']=localWin
    
    #plotTiming_train(infoToPlot, viz, win)

    
    return(win)
    
    
def plotTiming_train(infoToPlot, viz, win):
    
    time_train=infoToPlot['time_train']
    time_train=np.array(time_train)
    
    indexes=np.array(infoToPlot['id_batch_train'])

    localWin =viz.line(
       Y=time_train,
       X=indexes,
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='batch',
            ylabel='time',
            title='evolution of the time per batch in training mode',
            legend=['time_train'],
            ) ,win=win['time_train']
        )
        
    win['time_train']=localWin
    return(win)

def plotTiming_valid(infoToPlot, viz, win):

    time_valid=infoToPlot['time_valid']
    time_valid=np.array(time_valid)
    
    indexes=np.array(infoToPlot['id_batch_valid'])

    localWin =viz.line(
       Y=time_valid,
       X=indexes,
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='batch',
            ylabel='time',
            title='evolution of the time per batch in validation mode',
            legend=['time_valid'],
            ) ,win=win['time_valid']
        )
        
    win['time_valid']=localWin
    
    return(win)
    
def plotTiming_valid(data, viz, win):

    time_valid=infoToPlot['time_valid']
    time_valid=np.array(time_valid)
    
    indexes=np.array(infoToPlot['id_batch_valid'])

    localWin =viz.line(
       Y=time_valid,
       X=indexes,
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='batch',
            ylabel='time',
            title='evolution of the time per batch in validation mode',
            legend=['time_valid'],
            ) ,win=win['time_valid']
        )
        
    win['time_valid']=localWin
    
    return(win)
    
    
    
def plotValidPPL(dataValid, indexes, infoToPlot , viz, win):  
   

    dataValid=np.array(dataValid)
    indexes=np.array(indexes)

    localWin =viz.line(
       Y=dataValid,
       X=indexes,
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='batch',
            ylabel='Valid perplexity',
            title='evolution of the perplexity on validation set',
            legend=['valid Perp'],
            ) ,win=win['validPerp']
        )
        
    win['validPerp']=localWin
    
    #plotTiming_valid(infoToPlot, viz, win)
    
    return(win)
    
    
def plotMatching(infoToPlot , viz, win, mode='train'):  
   
   
    data=np.array(infoToPlot['matching_{}'.format(mode)])
    indexes=np.array(infoToPlot['id_batch_{}'.format(mode)])
    
   
    localWin =viz.line(
       Y=data,
       X=indexes,
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='batch',
            ylabel='number of matches per batches',
            title='evolution of the accuracy on the {} set'.format(mode),
            legend=['percentage of matches'],
            ) ,win=win['matching_{}'.format(mode)]
        )
        
    win['matching_{}'.format(mode)]=localWin
    
    #plotTiming_valid(infoToPlot, viz, win)
    
    return(win)

def plotSampledSentences(dataDict, viz, win):  
   

    input_warmup=dataDict[0]['input_warmup']    
    localWin =viz.text(input_warmup,
        opts=dict(
            showlegend=True,
            title='input warmup',
            ) ,win=win['input_warmup']
        )        
    win['input_warmup']=localWin
    
    
    output_warmup=dataDict[0]['output_warmup']    
    localWin =viz.text(output_warmup,
        opts=dict(
            showlegend=True,
            title='output warmup',
            ) ,win=win['output_warmup']
        )        
    win['output_warmup']=localWin
    
    input_sentence=dataDict[0]['input_sentence']    
    localWin =viz.text(input_sentence,
        opts=dict(
            showlegend=True,
            title='input sentence',
            ) ,win=win['input_sentence']
        )        
    win['input_sentence']=localWin
    
    
    output_sentence=dataDict[0]['output_sentence']    
    localWin =viz.text(output_sentence,
        opts=dict(
            showlegend=True,
            title='output sentence',
            ) ,win=win['output_sentence']
        )        
    win['output_sentence']=localWin
    
    
    return(win)

    
    
    
def visdom_plot(viz, win, infoToPlot, valid=False):    

    if valid==False:
        try:
            win=plotTrainPPL(infoToPlot['trainPerp'], infoToPlot['id_batch_train'], infoToPlot, viz, win)
        except:
            pass
    else: 
        try:
            win=plotValidPPL(infoToPlot['validPerp'], infoToPlot['id_batch_valid'], infoToPlot, viz, win)
        except:
            pass
    try:
        win=plotSampledSentences(infoToPlot['generated'], viz, win)
    except:
        pass
    
    try:
        win=plotMatching(infoToPlot, viz, win, mode='train')
    except:
        pass
    
    try:
        win=plotMatching(infoToPlot, viz, win, mode='valid')
    except:
        pass
    
    
    return (win)

