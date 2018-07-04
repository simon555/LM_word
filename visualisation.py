# -*- coding: utf-8 -*-


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pl
import numpy as np
matplotlib.rcParams.update({'font.size': 8})

from torchvision.utils import make_grid
from visdom import Visdom

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
    

def plotTrainPPL(dataTrain, viz, win):  
    NTrain=len(dataTrain)    
   

    dataTrain=np.array(dataTrain)
    
    localWin =viz.line(
       Y=dataTrain,
       X=np.linspace(1,NTrain,NTrain),
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
    
    
    return(win)
    
def plotValidPPL(dataValid, viz, win):  
    NValid=len(dataValid)    
   

    dataValid=np.array(dataValid)
    
    localWin =viz.line(
       Y=dataValid,
       X=np.linspace(1,NValid,NValid),
        opts=dict(
            showlegend=True,
            width=600,
            height=600,
            xlabel='epoch',
            ylabel='Valid perplexity',
            title='evolution of the perplexity on validation set',
            legend=['valid Perp'],
            ) ,win=win['validPerp']
        )
        
    win['validPerp']=localWin
    
    
    return(win)

def plotSampledSentences(dataDict, viz, win):  
   

    inputText=dataDict[0]['input_sentence']    
    localWin =viz.text(inputText,
        opts=dict(
            showlegend=True,
            title='input sentence',
            ) ,win=win['input_sentence']
        )
        
    win['input_sentence']=localWin
    
    expectedText=dataDict[0]['expected_sentence']    
    localWin =viz.text(expectedText,
        opts=dict(
            showlegend=True,
            title='expected sentence',
            ) ,win=win['expected_sentence']
        )
        
    win['expected_sentence']=localWin
    
    ouputText=dataDict[0]['output_sentence']    
    localWin =viz.text(ouputText,
        opts=dict(
            showlegend=True,
            title='output sentence',
            ) ,win=win['output_sentence']
        )
        
    win['output_sentence']=localWin
    
    
    return(win)
    
def visdom_plot(viz, win, infoToPlot, valid=False):    

    if valid==False:
        win=plotTrainPPL(infoToPlot['trainPerp'], viz, win)
    
    else: 
        win=plotValidPPL(infoToPlot['validPerp'], viz, win)

    
    win=plotSampledSentences(infoToPlot['generated'], viz, win)
    
    
    
    return (win)

