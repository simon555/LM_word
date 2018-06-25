import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, TimeDistributed, Activation
from keras.optimizers import Adam,SGD
import keras.backend as K
import numpy as np
import time

class LanguageModel(object):
    def __init__(self,params):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.sess = tf.Session(config = config)

        K.set_session(self.sess)
        # Pull out all of the parameters
        self.batch_size = params['batch_size']
        self.valid_batch_size = params['valid_batch_size']
        self.seq_len = params['seq_len']
        self.vocab_size = params['vocab_size']
        self.embed_size = params['embed_size']
        self.hidden_dim = params['hidden_dim']
        self.num_layers = params['num_layers']
        self.directoryOutLogs=params['directoryOutLogs']
        self.mode='train'
        self.LSTM=LSTM(units=self.hidden_dim, return_sequences=True, name='rnn_1', stateful=True)
        self.initializerDone=False
                


   
    
        
        
        with tf.device('/gpu:0'):
            # Set up the input placeholder
            self.input_seq = tf.placeholder(tf.float32, shape=[self.batch_size, self.seq_len])
         
            # Build the RNN
            self.rnn = Embedding(self.vocab_size, self.embed_size, input_length=self.seq_len, mask_zero=True)(self.input_seq)

        with tf.device('/gpu:1'):
            for l in range(self.num_layers):
                self.rnn = self.LSTM(self.rnn)
            rnn_output = tf.unstack(self.rnn, num=self.input_seq.shape[1], axis=1)
            self.w_proj = tf.Variable(tf.zeros([self.vocab_size, self.hidden_dim]))
            self.b_proj = tf.Variable(tf.zeros([self.vocab_size]))
            self.output_seq = tf.placeholder(tf.int64, shape=([None, self.seq_len]))
            losses = []
            outputs = []
            for t in range(self.seq_len):
                rnn_t = rnn_output[t]
                y_t = tf.reshape(self.output_seq[:, t],[-1,1])
                step_loss = tf.nn.sampled_softmax_loss(weights=self.w_proj, biases=self.b_proj, inputs=rnn_t,
                                                       labels=y_t, num_sampled=512, num_classes=self.vocab_size)
                losses.append(step_loss)
                outputs.append(tf.matmul(rnn_t, tf.transpose(self.w_proj)) + self.b_proj)
            self.step_losses = losses
            self.output = outputs
            self.loss = tf.reduce_mean(self.step_losses)
            self.softmax = tf.nn.softmax(self.output)
            
            
            
    def compile(self,lr=1e-3):
        self.loss_function = tf.reduce_mean(self.loss)
        #loss_scalar = tf.summary.scalar("loss",self.loss_function)
        self.opt = tf.train.AdamOptimizer(lr).minimize(self.loss_function)
        #self.train_writer = tf.summary.FileWriter( self.directoryOutLogs, self.sess.graph)

        self.sess.run(tf.initialize_all_variables())
        
        
    def train_on_batch(self,X_batch,Y_batch):
        #self.opt.run(session=self.sess,feed_dict={self.input_seq: X_batch, self.output_seq: Y_batch})

        self.reInitialize_LSTM_hidden()

        
        batch_time_length=len(X_batch[0])
        if batch_time_length % self.seq_len != 0 : 
            number_of_chunks=batch_time_length//self.seq_len +1
        else:
            number_of_chunks=batch_time_length//self.seq_len
        
        start_index=0
        end_index=min(batch_time_length,self.seq_len)
        loss_value=0
        #merged = tf.summary.merge_all()

        for chunk_no in range(number_of_chunks):
            X_input=X_batch[:,start_index:end_index]
            Y_input=Y_batch[:,start_index:end_index]

            _, tmp_loss_value = self.sess.run([self.opt, self.loss],feed_dict={self.input_seq: X_input, self.output_seq: Y_input})
            #self.train_writer.add_summary(summary)
            loss_value+=tmp_loss_value
            start_index=end_index
            end_index=min(batch_time_length, end_index + self.seq_len )
        
        
        return loss_value
    
    
    
    def predict(self,X,asarray=True):
        self.reInitialize_LSTM_hidden()

        preds = self.sess.run(self.softmax, feed_dict={self.input_seq: X})
        if asarray:
            preds = np.asarray(preds)
            ## Make dimensions more sensible ##
            preds = np.swapaxes(preds,0,1)
        return preds
    def evaluate(self,X,Y,normalize=False):
        #current_batch_size=X.shape[0]
        #paddings=np.array([[0,self.batch_size-current_batch_size],[0,0]])
        #X=np.pad(X,paddings, mode='constant')
        
        #t0 = time.time()
        self.reInitialize_LSTM_hidden()
        preds = self.sess.run(self.softmax, feed_dict={self.input_seq: X})
        #t1 = time.time()
        #print('time', t1-t0)

        preds = np.asarray(preds)
        preds = np.swapaxes(preds, 0, 1)
        #preds=preds[:self.valid_batch_size]
        
        log_prob=0
        n_tokens=0
        
        ## Note we're only going to use the non-zero entries ##
        for i in range(len(X)):
            for j in range(len(X[i])):
                if X[i,j] != 0:
                    correct_prob = preds[i,j,Y[i,j]]
                    log_prob += np.log(correct_prob)
                    n_tokens += 1.
      
        return log_prob, n_tokens
    
    
    
    def generate(self,seed='',temperature=1.0):
        pass
    def save(self,save_path='./'):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, save_path + 'model.ckpt')
        print("Model saved in file: %s" % save_path)

    def load(self,save_path='./'):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path + 'model.ckpt')
        print("Model restored.")# -*- coding: utf-8 -*-

    def reInitialize_LSTM_hidden(self):
        if self.LSTM.states is not [None, None]:
            if not self.initializerDone :
                self.init_hidden = tf.initialize_variables(self.LSTM.states)
                self.initializerDone=True
            self.sess.run(self.init_hidden)
        else:
            print('None state')



