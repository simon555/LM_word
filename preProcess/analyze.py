# -*- coding: utf-8 -*-


import os
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as pl
import numpy as np

#import nltk
#nltk.download('punkt')

#from nltk.tokenize import sent_tokenize, word_tokenize
              
import argparse
from tqdm import tqdm

#widgets = ['Text Analyze: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
#           ' ', ETA(), ' ', FileTransferSpeed()] #see docs for other options


parser = argparse.ArgumentParser()

parser.add_argument("-root", default = './data/splitted/smallData/train/', help="directory where the dataset lays", type=str)

parser.add_argument("-fileName", default="smallData", type=str)
parser.add_argument("-extension", default=".txt", type=str)

parser.add_argument("-Nsamples", default=100000, type=int)
parser.add_argument("-bins", default=50, type=int)
parser.add_argument("-Pkeep", default=0.5, type=float)

if os.name=='nt':
    default_path_to_save='./../stats/smallData/'
else:
    default_path_to_save='./../stats/springer_cui_tokenized/'
    
parser.add_argument("-path_to_save", default=default_path_to_save, type=str)


if os.name=='nt':
    default_path_to_TEXT='./../data/splitted/smallData/vocab/word/vocab_full_word.pickle'
else:
    default_path_to_TEXT=os.path.join('/mnt','raid1','text','big_files','splitted','springer_cui_tokenized','vocab','word','vocab_100_word.pickle')
    
parser.add_argument("-path_to_TEXT", default=default_path_to_TEXT, type=str)


args = parser.parse_args()
fileName=args.fileName+args.extension

root=args.root
outputDir='./stats/{}/'.format(args.fileName)

#if not os.path.exists(outputDir):
#    os.makedirs(outputDir)
    
    
Nsamples=args.Nsamples
bins=args.bins
p_keep=args.Pkeep    

endOfSentence= ['.', '!', '?']
fileText=os.path.join(root, fileName)


def get_stats(TEXT_path=args.path_to_TEXT,path_to_save=args.path_to_save):
    import dill as pickle
    TEXT=pickle.load(open(TEXT_path,'rb'))
    
    
    freqs=[]
    total_words=0
    for _,value in TEXT.vocab.freqs.items():
        freqs+=[value]
        total_words+=value
    
    freqs=-np.sort(-np.array(freqs))
    
    pl.clf()
    figure=pl.figure(0)
    pl.plot(freqs)
    pl.title('word appearance')
    pl.ylabel('#number of times')  
    pl.xlabel('#word index')
    pl.savefig(path_to_save + 'wordFreq.png') 
    
    check=[70,75,80,85,90,95,98,99, 99.5, 99.9]
    checks=iter(check)
    min_size_voc=[]
    
    thresh=next(checks)
    total=0
    local_number=0
    for value in tqdm(freqs):
        local_number+=1
        total+=value
        if total>thresh*0.01*total_words:
            min_size_voc+=[local_number]
            print('total seen ', total)
            print('percentage ', thresh*0.01*total_words)
            try:
                thresh=next(checks)
                print('new thresh ', thresh)
            except:
                break
    pl.clf()
    figure=pl.figure(1)
    pl.plot(check, min_size_voc)
    pl.title('min number of voc_size to match a % of the initial vocabulary')
    pl.xlabel('percentage of vocabulary')
    pl.ylabel('voc_size')
    pl.legend()
    pl.savefig(path_to_save + 'percentage_voc.png') 

    
    info='total size of corpus {} \n'.format(total_words)
    for i in range(len(check)):
        info+='% of voc : {} \t min voc_size : {} \n'.format(check[i], min_size_voc[i])
    print(info)
    with open(path_to_save + 'info_percentage_voc.txt','w') as file:
        file.write(info)
    
    


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
            dice=np.random.random()
            if dice>p_keep:
                continue
            else:
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
    pl.hist(numberOfCharPerWord,bins=bins)
    pl.title('stats per word')
    pl.ylabel('#char')    
    pl.savefig(outputDir + 'statsPerWord.png') 
    
    
    #plot the stats per sentence
    numberOfCharPerSentence=data['numberOfCharPerSentence']
    numberOfWordPerSentence=data['numberOfWordPerSentence']
    
    
    figure=pl.figure(1)
    pl.subplot(211)
    pl.hist(numberOfCharPerSentence, bins=bins)
    pl.title('stats per sentence')
    pl.ylabel('#char')   
    
    
    
    
    pl.subplot(212)
    pl.hist(numberOfWordPerSentence, bins=bins)
    pl.ylabel('#word')   
    
    pl.savefig(outputDir + 'statsPerSentence.png') 
    
   
    #plot the stats per article
    numberOfCharPerArticle=data['numberOfCharPerArticle']
    numberOfWordPerArticle=data['numberOfWordPerArticle']
    numberOfSentencePerArticle=data['numberOfSentencePerArticle']
    
    
    figure=pl.figure(2)
    pl.subplot(311)
    pl.hist(numberOfCharPerArticle, bins=bins)
    pl.title('stats per article')
    pl.ylabel('#char')   
    
    
    
    
    pl.subplot(312)
    pl.hist(numberOfWordPerArticle, bins=bins)
    pl.ylabel('#words')   
    
    pl.subplot(313)
    pl.hist(numberOfSentencePerArticle, bins=bins)
    pl.ylabel('#sentence') 
    
    pl.savefig(outputDir + 'statsPerArticle.png') 
    
    
    
#dataStats=analyze(fileText)   
#display()    
    



        
    
