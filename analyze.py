# -*- coding: utf-8 -*-


import os
import pickle
import matplotlib.pyplot as pl
import numpy as np

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from progressbar import *              
import argparse


widgets = ['Text Analyze: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options


parser = argparse.ArgumentParser()

parser.add_argument("-root", default = './data/train/', help="directory where the dataset lays", type=str)

parser.add_argument("-fileName", default="train", type=str)
parser.add_argument("-extension", default=".txt", type=str)

parser.add_argument("-Nsamples", default=1000, type=int)
parser.add_argument("-bins", default=50, type=int)



args = parser.parse_args()
fileName=args.fileName+args.extension

root=args.root
outputDir='./stats/{}/'.format(args.fileName)

if not os.path.exists(outputDir):
    os.makedirs(outputDir)
    
    
Nsamples=args.Nsamples
bins=args.bins
endOfSentence= ['.', '!', '?']
fileText=os.path.join(root, fileName)




def specialAppend(listOfValues, elt, Nsamples=Nsamples):
    '''
    used in the analyze function. Allows to store a small sample from the dataset when we iterate through it
    '''
    if len(listOfValues) <Nsamples:
        listOfValues.append(elt)
    else:
        index=np.random.randint(Nsamples)
        listOfValues[index]=elt
        


       


def analyze(fileText, outputDir=outputDir):
    '''
    Analyze the stats of the dataset, output samples using the special append
    '''
    pbar = ProgressBar(widgets=widgets)
    pbar.start()
    with open(fileText,'r+') as file:
        numberOfArticle=0
        numberOfSentences=0
        numberOfWords=0
        numberOfChar=0
        
        numberOfSentencePerArticle=[]
        meanNumberOfSentencePerArticle=0
        
        numberOfWordPerSentence=[]
        meanNumberOfWordPerSentence=0
        numberOfWordPerArticle=[]
        meanNumberOfWordPerArticle=0
        
        numberOfCharPerWord=[]
        meanNumberOfCharPerWord=0
        numberOfCharPerSentence=[]
        meanNumberOfCharPerSentence=0        
        numberOfCharPerArticle=[]
        meanNumberOfCharPerArticle=0
        
        for article in pbar(file.readlines()):
            numberOfArticle+=1                       
            sent_tokenize_list = sent_tokenize(article)
            specialAppend(numberOfSentencePerArticle,len(sent_tokenize_list))
            meanNumberOfSentencePerArticle = ( len(sent_tokenize_list) + meanNumberOfSentencePerArticle * (numberOfArticle - 1) ) / numberOfArticle
            
            tmp_numberOfWordPerArticle=0
            tmp_numberOfCharPerArticle=0
            
            for sentence in sent_tokenize_list:
                numberOfSentences+=1

                word_tokenize_list=word_tokenize(sentence)
                specialAppend(numberOfWordPerSentence,len(word_tokenize_list))
                meanNumberOfWordPerSentence = ( len(word_tokenize_list) + meanNumberOfWordPerSentence * (numberOfSentences - 1) ) / numberOfSentences

                tmp_numberOfWordPerArticle+=len(word_tokenize_list)
                
                tmp_numberOfCharPerSentence=0

                for word in word_tokenize_list:
                    #pbar.update(i) #this adds a little symbol at each iteration
                    
                    numberOfWords+=1
                    numberOfChar+=len(word)
                    meanNumberOfCharPerWord= ( len(word) + meanNumberOfCharPerWord * (numberOfWords - 1) ) / numberOfWords

                    specialAppend(numberOfCharPerWord,len(word))
                    tmp_numberOfCharPerSentence+=len(word)
                    tmp_numberOfCharPerArticle+=len(word)
                specialAppend(numberOfCharPerSentence,tmp_numberOfCharPerSentence)
                meanNumberOfCharPerSentence= ( tmp_numberOfCharPerSentence + meanNumberOfCharPerSentence * (numberOfSentences - 1) ) / numberOfSentences

                
            specialAppend(numberOfWordPerArticle,tmp_numberOfWordPerArticle)
            meanNumberOfWordPerArticle = ( tmp_numberOfWordPerArticle + meanNumberOfWordPerArticle * (numberOfArticle - 1) ) / numberOfArticle

        
            specialAppend(numberOfCharPerArticle,tmp_numberOfCharPerArticle)
            meanNumberOfCharPerArticle = ( tmp_numberOfCharPerArticle + meanNumberOfCharPerArticle * (numberOfArticle - 1) ) / numberOfArticle
            
    
    stats={'numberOfArticles':numberOfArticle,
            'numberOfSentences':numberOfSentences,
            'numberOfWords':numberOfWords,
            'numberOfChars':numberOfChar,
            'numberOfCharPerWord':numberOfCharPerWord,
            'numberOfCharPerSentence':numberOfCharPerSentence, 
            'numberOfCharPerArticle':numberOfCharPerArticle, 
            'numberOfWordPerSentence':numberOfWordPerSentence, 
            'numberOfWordPerArticle':numberOfWordPerArticle, 
            'numberOfSentencePerArticle':numberOfSentencePerArticle, 
            'meanNumberOfSentencePerArticle':meanNumberOfSentencePerArticle,
            'meanNumberOfWordPerSentence':meanNumberOfWordPerSentence,
            'meanNumberOfWordPerArticle':meanNumberOfWordPerArticle,
            'meanNumberOfCharPerWord':meanNumberOfCharPerWord,
            'meanNumberOfCharPerSentence':meanNumberOfCharPerSentence,
            'meanNumberOfCharPerArticle':meanNumberOfCharPerArticle}
     
     
    with open( outputDir +'dataStats.pkl', 'wb') as f:
        pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)
#    print(meanNumberOfCharPerWord)
#    print(meanNumberOfCharPerSentence)
#    print(meanNumberOfCharPerArticle)
#    
#    print(meanNumberOfWordPerSentence)
#    print(meanNumberOfWordPerArticle)
#    
#    print(meanNumberOfSentencePerArticle)

    

    return(stats)



def load_obj(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def display(dataFile=outputDir + 'dataStats.pkl' , outputDir=outputDir, bins=bins):
    
    data=load_obj(dataFile)
    
    
    #save the global data in a txt file
    with open(outputDir + 'globalInfo.txt', 'w') as outstream:
            outstream.write('number of articles' + '\t' + str(data['numberOfArticles']) + '\n')   
            outstream.write('number of sentences' + '\t' + str(data['numberOfSentences']) + '\n') 
            outstream.write('number of words' + '\t' + str(data['numberOfWords']) + '\n') 
            outstream.write('number of chars' + '\t' +  str(data['numberOfChars']) + '\n \n') 
            
            outstream.write('mean number of chars per word' + '\t' +  str(data['meanNumberOfCharPerWord']) + '\n') 
            outstream.write('mean number of chars per sentence' + '\t' +  str(data['meanNumberOfCharPerSentence']) + '\n') 
            outstream.write('mean number of chars per article' + '\t' +  str(data['meanNumberOfCharPerArticle']) + '\n \n') 
            
            outstream.write('mean number of word per sentence' + '\t' +  str(data['meanNumberOfWordPerSentence']) + '\n') 
            outstream.write('mean number of word per article' + '\t' +  str(data['meanNumberOfWordPerArticle']) + '\n \n') 
            
            outstream.write('mean number of sentence per article' + '\t' +  str(data['meanNumberOfSentencePerArticle']) + '\n') 

            
            
    
    #plot the stats per word
    numberOfCharPerWord=data['numberOfCharPerWord']

    figure=pl.figure(0)
    pl.hist(numberOfCharPerWord)
    pl.title('number of characters per word')
    pl.xlabel('word occurence')
    pl.ylabel('number of character')    
    pl.savefig(outputDir + 'statsPerWord.png') 
    
    
    #plot the stats per sentence
    numberOfCharPerSentence=data['numberOfCharPerSentence']
    numberOfWordPerSentence=data['numberOfWordPerSentence']
    
    
    figure=pl.figure(1)
    pl.subplot(211)
    pl.hist(numberOfCharPerSentence, bins=bins)
    pl.title('stats per sentence')
    pl.xlabel('sentence occurence')
    pl.ylabel('number of character')   
    
    
    
    
    pl.subplot(212)
    pl.hist(numberOfWordPerSentence, bins=bins)
    pl.xlabel('sentence occurence')
    pl.ylabel('number of words')   
    
    pl.savefig(outputDir + 'statsPerSentence.png') 
    
   
    #plot the stats per article
    numberOfCharPerArticle=data['numberOfCharPerArticle']
    numberOfWordPerArticle=data['numberOfWordPerArticle']
    numberOfSentencePerArticle=data['numberOfSentencePerArticle']
    
    
    figure=pl.figure(2)
    pl.subplot(311)
    pl.hist(numberOfCharPerArticle, bins=bins)
    pl.title('stats per article')
    pl.xlabel('article occurence')
    pl.ylabel('#character')   
    
    
    
    
    pl.subplot(312)
    pl.hist(numberOfWordPerArticle, bins=bins)
    pl.xlabel('sentence occurence')
    pl.ylabel('#words')   
    
    pl.subplot(313)
    pl.hist(numberOfSentencePerArticle, bins=bins)
    pl.xlabel('sentence occurence')
    pl.ylabel('#sentences') 
    
    pl.savefig(outputDir + 'statsPerArticle.png') 
    
    
    
dataStats=analyze(fileText)   
display()    
    
                
            
        
     


        
    
