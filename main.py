from __future__ import division, print_function, unicode_literals
import argparse
from pathlib import Path
import h5py
import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from loss import spread_loss, cross_entropy, margin_loss
from network import baseline_model_kimcnn, baseline_model_cnn, capsule_model_A, capsule_model_B
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
BIG_CATEGORY = 'beauty'
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
psychic_learners_dir = Path.cwd().parent / 'PsychicLearners'
TRAIN_CSV = str(psychic_learners_dir / 'data' / 'csvs' /
                f'{BIG_CATEGORY}_train_split.csv')
VALID_CSV = str(psychic_learners_dir / 'data' / 'csvs' /
                f'{BIG_CATEGORY}_valid_split.csv')
TEST_CSV = str(psychic_learners_dir / 'data' / 'csvs' /
               f'{BIG_CATEGORY}_test_split.csv')

tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)
#tf.logging.set_verbosity(tf.logging.FATAL)
parser = argparse.ArgumentParser()

parser.add_argument('--embedding_type', type=str, default='static',
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--dataset', type=str, default='reuters_multilabel_dataset',
                    help='Options: reuters_multilabel_dataset, MR_dataset, SST_dataset')

parser.add_argument('--loss_type', type=str, default='margin_loss',
                    help='margin_loss, spread_loss, cross_entropy')

parser.add_argument('--model_type', type=str, default='capsule-B',
                    help='CNN, KIMCNN, capsule-A, capsule-B')

parser.add_argument('--has_test', type=int, default=1, help='If data has test, we use it. Otherwise, we use CV on folds')    
parser.add_argument('--has_dev', type=int, default=1, help='If data has dev, we use it, otherwise we split from train')    

parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=25, help='Batch size for training')

parser.add_argument('--use_orphan', type=bool, default='True', help='Add orphan capsule or not')
parser.add_argument('--use_leaky', type=bool, default='False', help='Use leaky-softmax or not')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')#CNN 0.0005 
parser.add_argument('--margin', type=float, default=0.2, help='the initial value for spread loss')
parser.add_argument('--big_cat', type=str, default='beauty', help='fashion/mobile/beauty')

import json
args = parser.parse_args()
params = vars(args)
print(json.dumps(params, indent = 2))


def load_data(dataset):
    
    f = h5py.File(dataset+'.hdf5', 'r') 
    print('loading data...')    
    print(dataset)
    print("Keys: %s" % f.keys())
    
    train_df = pd.read_csv(TRAIN_CSV)
    valid_df = pd.read_csv(VALID_CSV)
    test_df = pd.read_csv(TEST_CSV)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_df['title'].values)
    train_titles_seq = tokenizer.texts_to_sequences(train_df['title'].values)
    valid_titles_seq = tokenizer.texts_to_sequences(valid_df['title'].values)
    test_titles_seq = tokenizer.texts_to_sequences(test_df['title'].values)
    train_titles_seq = pad_sequences(train_titles_seq, padding='post', maxlen=16)
    valid_titles_seq = pad_sequences(valid_titles_seq, padding='post', maxlen=16)
    test_titles_seq = pad_sequences(test_titles_seq, padding='post', maxlen=16)
    vocab_size = len(tokenizer.word_index) + 1

    train_label = train_df['Category'].values
    valid_label = valid_df['Category'].values
    if args.big_cat == 'fashion':
        train_label = train_label - 17
        valid_label = valid_label - 17
    elif args.big_cat == 'mobile':
        train_label = train_label - 31
        valid_label = valid_label - 31
    test_label = np.zeros(len(test_titles_seq))

    w2v = list(f['w2v'])
    print('TRAIN _label:', train_label)
    if args.use_orphan:
        args.num_classes = len(np.unique(train_label))

    return train_titles_seq, train_label, test_titles_seq, test_label, valid_titles_seq, valid_label, vocab_size

class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, dataset,label, batch_size, input_size, is_shuffle=True):
      self._dataset = dataset
      self._label = label
      self._batch_size = batch_size    
      self._cursor = 0      
      self._input_size = input_size      
      
      if is_shuffle:
          index = np.arange(len(self._dataset))
          np.random.shuffle(index)
          self._dataset = np.array(self._dataset)[index]
          self._label = np.array(self._label)[index]
      else:
          self._dataset = np.array(self._dataset)
          self._label = np.array(self._label)
    def next(self):
      if self._cursor + self._batch_size > len(self._dataset):
          self._cursor = 0
      """Generate a single batch from the current cursor position in the data."""      
      batch_x = self._dataset[self._cursor : self._cursor + self._batch_size,:]
      batch_y = self._label[self._cursor : self._cursor + self._batch_size]
      self._cursor += self._batch_size
      return batch_x, batch_y

train, train_label, test, test_label, dev, dev_label, vocab_size = load_data(args.dataset)    

args.vocab_size = vocab_size
args.vec_size = 128
args.max_sent = len(train[0])
print('max sent: ', args.max_sent)
print('vocab size: ', args.vocab_size)
print('vec size: ', args.vec_size)
print('num_classes: ', args.num_classes)

train, train_label = shuffle(train, train_label)

with tf.device('/cpu:0'):
    global_step = tf.train.get_or_create_global_step()

label = ['-1', 'earn', 'money-fx', 'trade', 'acq', 'grain', 'interest', 'crude', 'ship']
label = map(str,label)
args.max_sent = 16
threshold = 0.5
with tf.device('/gpu:0'):
    X = tf.placeholder(tf.int32, [args.batch_size, args.max_sent], name="input_x")
    y = tf.placeholder(tf.int64, [args.batch_size, int(args.num_classes)], name="input_y")
    is_training = tf.placeholder_with_default(False, shape=())    
    learning_rate = tf.placeholder(dtype='float32') 
    margin = tf.placeholder(shape=(),dtype='float32') 

    l2_loss = tf.constant(0.0)

    #w2v = np.array(w2v,dtype=np.float32)
    if args.embedding_type == 'rand':
        W1 = tf.Variable(tf.random_uniform([tf.float32(args.vocab_size), tf.float32(args.vec_size)], -0.25, 0.25),name="Wemb")
        X_embedding = tf.nn.embedding_lookup(W1, X)
        X_embedding = X_embedding[...,tf.newaxis] 
    if args.embedding_type == 'static':
        W1 = tf.Variable(w2v, trainable = False)
        X_embedding = tf.nn.embedding_lookup(W1, X)
        X_embedding = X_embedding[...,tf.newaxis] 
    if args.embedding_type == 'nonstatic':
        W1 = tf.Variable(w2v, trainable = True)
        X_embedding = tf.nn.embedding_lookup(W1, X)
        X_embedding = X_embedding[...,tf.newaxis] 
    if args.embedding_type == 'multi-channel':
        W1 = tf.Variable(w2v, trainable = True)
        W2 = tf.Variable(w2v, trainable = False)
        X_1 = tf.nn.embedding_lookup(W1, X)
        X_2 = tf.nn.embedding_lookup(W2, X) 
        X_1 = X_1[...,tf.newaxis]
        X_2 = X_2[...,tf.newaxis]
        X_embedding = tf.concat([X_1,X_2],axis=-1)

    tf.logging.info("input dimension:{}".format(X_embedding.get_shape()))

    if args.model_type == 'capsule-A':    
        poses, activations = capsule_model_A(X_embedding, int(args.num_classes))
    if args.model_type == 'capsule-B':    
        poses, activations = capsule_model_B(X_embedding, int(args.num_classes))
    if args.model_type == 'CNN':    
        poses, activations = baseline_model_cnn(X_embedding, int(args.num_classes))
    if args.model_type == 'KIMCNN':    
        poses, activations = baseline_model_kimcnn(X_embedding, args.max_sent, int(args.num_classes))
        
    if args.loss_type == 'spread_loss':
        loss = spread_loss(y, activations, margin)
    if args.loss_type == 'margin_loss':    
        loss = margin_loss(y, activations)
    if args.loss_type == 'cross_entropy':
        loss = cross_entropy(y, activations)

    y_pred = tf.argmax(activations, axis=1, name="y_proba")    
    correct = tf.equal(tf.argmax(y, axis=1), y_pred, name="correct")
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss, name="training_op")
    gradients, variables = zip(*optimizer.compute_gradients(loss))

    grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
                for g in gradients if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]
    with tf.control_dependencies(grad_check):
        training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)      

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
#sess = tf.Session(config=config)
from keras import utils

n_iterations_per_epoch = len(train) // args.batch_size
n_iterations_test = len(test) // args.batch_size
n_iterations_dev = len(dev) // args.batch_size    

mr_train = BatchGenerator(train,train_label, args.batch_size, 0)    
mr_dev = BatchGenerator(dev,dev_label, args.batch_size, 0)
mr_test = BatchGenerator(test,test_label, args.batch_size, 0, is_shuffle=False)

best_model = None
best_epoch = 0
best_acc_val = 0.

init = tf.global_variables_initializer()
sess.run(init)     

lr = args.learning_rate
m = args.margin
max_acc = 0
saver = tf.train.Saver(tf.global_variables())
os.makedirs(os.path.join(BIG_CATEGORY, args.model_type), exist_ok=True)
for epoch in range(args.num_epochs):
    for iteration in range(1, n_iterations_per_epoch + 1):                
        X_batch, y_batch = mr_train.next()     
        y_batch = utils.to_categorical(y_batch, args.num_classes)        
        _, loss_train, probs, capsule_pose = sess.run(
            [training_op, loss, activations, poses],
            feed_dict={X: X_batch[:,:args.max_sent],
                       y: y_batch,
                       is_training: True,
                       learning_rate:lr,
                       margin:m})        
        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                  iteration, n_iterations_per_epoch,
                  iteration * 100 / n_iterations_per_epoch,
                  loss_train),
              end="")
    loss_vals, acc_vals = [], []
    for iteration in range(1, n_iterations_dev + 1):
        X_batch, y_batch = mr_dev.next()            
        y_batch = utils.to_categorical(y_batch, args.num_classes)
        loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch[:,:args.max_sent],
                           y: y_batch,
                           is_training: False,
                           margin:m})
        loss_vals.append(loss_val)
        acc_vals.append(acc_val)
    loss_val, acc_val = np.mean(loss_vals), np.mean(acc_vals)    
    print("\rEpoch: {}  Val accuracy: {:.1f}%  Loss: {:.4f}".format(
        epoch + 1, acc_val * 100, loss_val))
    if acc_val > max_acc:
        max_acc = acc_val
        
        saver.save(sess, "{}/{}/{}.ckpt".format(BIG_CATEGORY,args.model_type, acc_val), global_step=epoch)
        print("Model is saved.\n")

    preds_list, y_list = [], []
    """
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = mr_test.next()             
        probs = sess.run([activations],
                feed_dict={X:X_batch[:,:args.max_sent],
                           is_training: False})
        preds_list = preds_list + probs[0].tolist()
        y_list = y_list + y_batch.tolist()
        
    y_list = np.array(y_list)
    preds_probs = np.array(preds_list)                
    preds_probs[np.where( preds_probs >= threshold )] = 1.0
    preds_probs[np.where( preds_probs < threshold )] = 0.0 
    acc = accuracy_score(y_list, preds_probs)
    print('\rER: %.3f' % acc) """

    if args.model_type == 'CNN' or args.model_type == 'KIMCNN':
        lr = max(1e-6, lr * 0.95)
    if args.loss_type == 'margin_loss':    
        m = min(0.9, m + 0.1)
