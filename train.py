from data_utils import Dataset
from keras.utils import generic_utils
from model import LanguageModel
import numpy as np
import tensorflow as tf
import time as time
import os

tf.logging.set_verbosity(tf.logging.ERROR)


if os.name=='nt':
    #running on my local windows machine for debug
    data_dir = './data/train/'
    valid_data_dir = './data/valid/'
else:
    #running on GPU server
    data_dir = '/mnt/raid1/text/big_files/train/'
    valid_data_dir = '/mnt/raid1/text/big_files/valid/'
    
    
save_dir = './checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


num_words = None

seq_len = 25
batch_size = 192
valid_batch_size = 16 ## Needs to be smaller due to memory issues
embed_size = 128
num_epochs = 20
hidden_size = 256
num_layers = 1

dataset = Dataset(data_dir,num_words)
dataset.set_batch_size(batch_size)
dataset.set_seq_len(seq_len)
dataset.save('./checkpoints/')

params = {}
params['vocab_size'] = dataset.vocab_size
params['num_classes'] = dataset.vocab_size
params['batch_size'] = batch_size
params['seq_len'] = seq_len
params['hidden_dim'] = hidden_size
params['num_layers'] = num_layers
params['embed_size'] = embed_size

model = LanguageModel(params)
model.compile()
eval_softmax = 5
for epoch in range(num_epochs):
    dataset.set_data_dir(data_dir)
    dataset.set_batch_size(batch_size)
    progbar = generic_utils.Progbar(dataset.token.document_count)
    for X_batch,Y_batch in dataset:
        t0 = time.time()
        loss = model.train_on_batch(X_batch,Y_batch)
        perp = np.exp(np.float32(loss))
        t1 = time.time()
        wps = np.round((batch_size * seq_len)/(t1-t0))
        progbar.add(len(X_batch), values=[("loss", loss),("perplexity", perp),("words/sec", wps)])
    model.save(save_dir)
    dataset.set_data_dir(valid_data_dir)
    dataset.set_batch_size(valid_batch_size)
    valid_logprob = 0.
    tokens = 0.
    count = 0
    if epoch % eval_softmax == 0:
        print ('\n\nEstimating validation perplexity...')
        if epoch == 0:
            n_valid_batches = 0
        else:
            progbar = generic_utils.Progbar(n_valid_batches)
        for X_batch, Y_batch in dataset:
            if epoch == 0:
                n_valid_batches += 1
            else:
                progbar.add(1)
            log_prob, n_tokens = model.evaluate(X_batch, Y_batch)
            count += 1
            valid_logprob += log_prob
            tokens += n_tokens
        valid_perp = np.exp(-valid_logprob/tokens)
        print ('\nValidation Perplexity: ' + str(valid_perp) + '\n')