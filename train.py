from data_utils import Dataset
from keras.utils import generic_utils
from model import LanguageModel
import numpy as np
import tensorflow as tf
import time as time
import os
from progressbar import Percentage, Bar, FileTransferSpeed, ETA, ProgressBar

tf.logging.set_verbosity(tf.logging.ERROR)


if os.name=='nt':
    #running on my local windows machine for debug
    data_dir = './data/splitted/smallData/valid/'
    valid_data_dir = './data/splitted/smallData/test/'
    datasetName='smallData'
else:
    #running on GPU server
    data_dir = '/mnt/raid1/text/big_files/splitted/springer_cui_tokenized/train/'
    valid_data_dir = '/mnt/raid1/text/big_files/splitted/springer_cui_tokenized/valid/'
    datasetName='springer_cui_tokenized'
    

dataset_specific_info='./stats/{}/'.format(datasetName)

ExpName='LM_Model'
Nexperience=1

directoryOut='./results/{}_{}_Exp{}/'.format(ExpName,datasetName,Nexperience)
    
if not os.path.exists(directoryOut):
    print('new directory : ',directoryOut)        
else:
    while(os.path.exists(directoryOut)):
        print('directory already exists : ',directoryOut)
        Nexperience+=1
        directoryOut='./results/{}_{}_Exp{}/'.format(ExpName,datasetName,Nexperience)
    print('new directory : ',directoryOut)
        
directoryCkpt=directoryOut+'checkpoint/'
directoryData=directoryOut+'data/'
    

os.makedirs(directoryOut) 
os.makedirs(directoryData)
os.makedirs(directoryCkpt)

num_words = None

seq_len = 25
batch_size = 64
valid_batch_size = 64 ## Needs to be smaller due to memory issues
embed_size = 128
num_epochs = 20
hidden_size = 256
num_layers = 1

dataset = Dataset(data_dir,num_words)
dataset.set_batch_size(batch_size)
dataset.set_seq_len(seq_len)
dataset.save(dataset_specific_info)

params = {}
params['vocab_size'] = dataset.vocab_size
params['num_classes'] = dataset.vocab_size
params['batch_size'] = batch_size
params['valid_batch_size'] = valid_batch_size
params['seq_len'] = seq_len
params['hidden_dim'] = hidden_size
params['num_layers'] = num_layers
params['embed_size'] = embed_size

model = LanguageModel(params)
model.compile()
eval_softmax = 5

total_time_training=0
total_time_valid=0
loss_list=''
perp_list=''
wps_list='' 

time_per_batch=''
time_per_epoch=''


for epoch in range(num_epochs):
    dataset.set_data_dir(data_dir)
    dataset.set_batch_size(batch_size)
    progbar = generic_utils.Progbar(dataset.token.document_count)
    t_epoch_start=time.time()
       
    for X_batch,Y_batch in dataset:
#        if X_batch.shape[0]<batch_size:
#            print('early stop batch size : ', X_batch.shape[0])
#            continue
        
        t0 = time.time()
        loss = model.train_on_batch(X_batch,Y_batch)
        loss_list+='{} \n'.format(loss)
        perp = np.exp(np.float32(loss))
        perp_list+='{} \n'.format(perp)
        t1 = time.time()
        
        time_per_batch+='{} \n'.format(t1-t0)
        
        wps = np.round((batch_size * seq_len)/(t1-t0))
        wps_list+='{} \n'.format(wps)
        
        progbar.add(len(X_batch), values=[("loss", loss),("perplexity", perp),("words/sec", wps)])
    t_epoch_end=time.time()
    total_epoch=t_epoch_end-t_epoch_start
    time_per_epoch+='{} \n'.format(total_epoch)

    
    total_time_training+=total_epoch
    
    model.save(directoryCkpt)
    
    print('save epoch stats...')
    
    with open(directoryData + 'loss.txt', 'a') as outstream:
        outstream.write(loss_list) 
    with open(directoryData + 'perp.txt', 'a') as outstream:
        outstream.write(perp_list) 
    with open(directoryData + 'wps.txt', 'a') as outstream:
        outstream.write(wps_list) 
    with open(directoryData + 'time_per_batch.txt', 'a') as outstream:
        outstream.write(time_per_batch) 
    with open(directoryData + 'time_per_epoch.txt', 'a') as outstream:
        outstream.write(time_per_epoch) 
        
    loss_list=''
    perp_list=''
    wps_list=''  
    time_per_batch=''
    time_per_epoch='' 
        
    print('done')
    
    
    
    dataset.set_data_dir(valid_data_dir)
    dataset.set_batch_size(valid_batch_size)
    valid_logprob = 0.
    tokens = 0.
    count = 0
    if epoch % eval_softmax == 0:
        print ('\n\nEstimating validation perplexity...')
        t_valid_start=time.time()

        if epoch == 0:
            n_valid_batches = 0
        
        widgets = ['Validation perplexity....: ', Percentage(), ' ', Bar(marker='0',left='[',right=']'),
           ' ', ETA(), ' ', FileTransferSpeed()]
        pbar = ProgressBar(widgets=widgets)
        pbar.start()
        for X_batch, Y_batch in pbar(dataset):
            if epoch == 0:
                n_valid_batches += 1
            
            log_prob, n_tokens = model.evaluate(X_batch, Y_batch)
            count += 1
            valid_logprob += log_prob
            tokens += n_tokens
        valid_perp = np.exp(-valid_logprob/tokens)
        t_valid_end=time.time()
        t_valid_batch=t_valid_end-t_valid_start
        total_time_valid+=t_valid_batch
        print ('\nValidation Perplexity: ' + str(valid_perp) + '\n')


total_time=total_time_training+total_time_valid

with open(directoryData + 'timeInfo.txt', 'a') as outstream:
    outstream.write('total time of training : {} \n'.format(total_time_training))
    outstream.write('total time of validation : {} \n'.format(total_time_valid))
    outstream.write('total time : {} \n'.format(total_time))

print('done')
print('total time : ', total_time)