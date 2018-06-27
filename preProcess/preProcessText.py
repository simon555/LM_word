# -*- coding: utf-8 -*-

import csv
import os



from progressbar import *  
import fileinput
import string
from tqdm import tqdm




if os.name=='nt':
    inputDir='./data/clean/'
    DatasetName='smallData'
    fileName='clean.txt'
    outputDir='./data/splitted/{}/'.format(DatasetName)
    
else:
    inputDir='/mnt/raid1/text/big_files/'
    DatasetName='springer_cui_tokenized'
    fileName='    .txt'
    outputDir='/mnt/raid1/text/big_files/splitted/{}/'.format(DatasetName)
 
    
#infoDir='./stats/{}/'.format(DatasetName)
#fileText= inputDir + fileName
#
#outputDir_train = outputDir + 'train/'
#outputDir_valid = outputDir + 'valid/'
#outputDir_test = outputDir + 'test/'
#
#to_make=[infoDir, outputDir_train, outputDir_valid, outputDir_test]
#
#for directory in to_make:
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    
#
    
input_folder="C://Users//simon//Desktop//HMS//projects//LM_word//data//splitted//smallData//"

#fileText=os.path.join(inputDir, fileName)
def preProcess(input_folder=input_folder, output_folder=input_folder):
    train_data=input_folder + 'train.txt'
    test_data=input_folder + 'test.txt'
    valid_data=input_folder + 'valid.txt'
    
    
    tr = str.maketrans("", "", string.punctuation)
    
    with open(valid_data, 'r') as input_file:
        for line in tqdm(input_file.readlines()):
            new_line=line.translate(tr).lower()
            with open(input_folder + 'valid_noPunc.txt','a') as output_file:
                output_file.write(new_line)
                
    with open(test_data, 'r') as input_file:
        for line in tqdm(input_file.readlines()):
            new_line=line.translate(tr).lower()
            with open(input_folder + 'test_noPunc.txt','a') as output_file:
                output_file.write(new_line)
                
    with open(train_data, 'r') as input_file:
        for line in tqdm(input_file.readlines()):
            new_line=line.translate(tr).lower()
            with open(input_folder + 'train_noPunc.txt','a') as output_file:
                output_file.write(new_line)
    
    print('do not forget to remove the old not processed files and rename the new oness!')
    
#    if os.name=='nt':
#        delete='del'
#    else:
#        delete='rm'
#    
#    command= delete + ' ' + input_folder + 'train.txt'
#    print(os.system(command))
#
#    command= delete + ' ' + input_folder +'test.txt'
#    print(os.system(command))
#
#    command= delete + ' ' + input_folder +'valid.txt'
#    print(os.system(command))
#
#    
#    if os.name=='nt':
#        rename='rename'
#    else:
#        rename='mv'
#    
#    command= rename + " " + input_folder + 'train_noPunc.txt train.txt'
#    print(os.system(command))
#
#    command= rename + " " + input_folder + 'valid_noPunc.txt valid.txt'
#    print(os.system(command))
#
#    command= rename + " " + input_folder + 'test_noPunc.txt test.txt'
#    print(os.system(command))
#
#    
#
#    print('done : ', check==0)


        

   
def splitData(fileText='C://Users\simon\Desktop\HMS\projects\LM_word\data\clean//clean.txt'):
    '''
    split the dataset into 3 sets : train, valid and test
    '''
    
    #count the number of lines, and compute the indexes of the start/end of train/valid/test dataset
    with open(fileText) as file:
        widgets = ['counting lines....: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        pbar.start()
        i=0
        for line in pbar(file.readlines()):
            i+=1
        NumberOfLines=i
        #with open(infoDir + 'line_count.txt', 'a') as outstream:
        #        outstream.write('number of total lines : {}'.format( NumberOfLines) + '\n') 
           
        trainMaxIndex=int(0.7*NumberOfLines)
        validMaxIndex=trainMaxIndex + int(0.2*NumberOfLines)
        
     
    #split the dataset according to 70% training / 20% validation / 10% test
    
    with open(fileText) as file:
        i=0
        training_lines=0
        valid_lines=0
        test_lines=0
        widgets = ['splitting the data....: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        pbar.start()
        for line in pbar(file.readlines()):
            if i<trainMaxIndex:
                training_lines+=1
                with open(outputDir + 'train.txt', 'a') as outstream:
                    outstream.write(line) 
            elif trainMaxIndex<i and i < validMaxIndex :
                valid_lines+=1
                with open(outputDir + 'valid.txt', 'a') as outstream:
                    outstream.write(line) 
                    
            elif validMaxIndex < i :
                test_lines+=1
                with open(outputDir + 'test.txt', 'a') as outstream:
                    outstream.write(line)            
            i+=1
        with open(infoDir + 'line_count.txt', 'a') as outstream:
            outstream.write('number of training lines : {}'.format( training_lines) + '\n') 
            outstream.write('number of validation lines : {}'.format( valid_lines) + '\n') 
            outstream.write('number of test lines : {}'.format( test_lines) + '\n') 
        


     

#splitData()
        
    

def cleanData(fileText):
    '''
    clean the dataset (remove the labels) 
    works for the smallm dataset on my local windows machine
    '''
    with open(fileText) as tsvfile:
        reader = csv.DictReader(tsvfile, dialect='excel-tab')
        for row in reader:
            article = row['EssayText']               
            with open(outputDir + 'clean/', 'a') as outstream:
                outstream.write(article + '\n')   
                
                
def build_final_dataset():
    '''
    build the vocab of the model based on the training set
    modify the valid and test set, including a <unk> character
    '''

    return(0)
    
