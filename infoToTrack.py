# -*- coding: utf-8 -*-
import pickle
import os


def getInfo():
    infoToPlot={'trainPerp':[],
            'id_batch_train':[],
            'time_train':[],
            'validPerp':[],
            'id_batch_valid':[],
            'time_valid':[],
            'generated':[],
            'matching_train':[],
            'matching_valid':[]}
    	
    return(infoToPlot)
    

def updateDataStorage(infoToPlot, directory):
    """
    fill the data storage with the temp infoToPlot dictionnary
    then this temp dict is reset
    """
    
    
    dataStored=pickle.load(open(os.path.join(directory,'data.pkl'), 'rb'))
    
    for key, value in dataStored.items():
        value += infoToPlot[key]
        infoToPlot[key]=[]             
    
    
    with open(os.path.join(directory,'data.pkl'), 'wb') as f:
                    pickle.dump(dataStored, f, pickle.HIGHEST_PROTOCOL)