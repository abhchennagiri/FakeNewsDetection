#!/usr/bin/env python
# import the required packages here
import tensorflow as tf
import numpy as np
from tensorflow.contrib import learn
import os
from collections import OrderedDict
from models.cnn import CNN
from models.hybrid_cnn import Hybrid_CNN
import time, datetime
import re
import statistics

train_file = "train.tsv"
test_file = 'test.tsv'
valid_file = 'valid.tsv'
embed_file = 'wiki.en.vec'
embed_file = "glove.6B.50d.txt"
embed_file = "glove.6B.200d.txt"
embed_file = "glove.6B.300d.txt"
output_file = "predictions.txt"
train = True
#train = False
test = False
test = True

def run(train_file, valid_file, test_file, output_file):
        #Specify the hyperparameters here
        dim = 300
        dropout_keep_prob = 0.6
        num_filters = 128
        filter_sizes = [3,4,5]
        num_epochs = 10
        batch_size = 64
        md_f_sizes = [3,4]
        md_num_filters = 20

        x_text, y, _, metadata, md_len = load_data_and_labels( train_file )
        #print ( md_len, len( md_len ) )
        #print( len(metadata) )
        #return 
        x_dev, y_dev, _, md_dev, md_dev_len = load_data_and_labels( valid_file )
        x_test, md_test, md_len_test = load_data_and_labels( test_file, test=True  )
        x, vocab_processor = build_vocab( x_text, x_dev, x_test, metadata )
        vocab_size = len( vocab_processor.vocabulary_)
        md = np.array(list(vocab_processor.fit_transform(metadata))) 
        #x_shuf, y_shuf, md_shuf, md_len_shuf = shuffle_data( x, y, md, md_len )
        #x_train, x_dev, y_train, y_dev = split_data( x_shuf, y_shuf, y )
        pretrained_embeddings = load_embeddings( embed_file, vocab_processor, dim )
        #x_train, y_train, m_train, m_len_train = x_shuf, y_shuf, md_shuf, md_len_shuf
        x_train, y_train, m_train, m_len_train = np.asarray(x), np.asarray(y), np.asarray(md), np.asarray(md_len)
        x_dev = np.array(list(vocab_processor.fit_transform(x_dev)))
        m_dev = np.array(list(vocab_processor.fit_transform(md_dev)))
        if train:
            with tf.Graph().as_default():
                sess = tf.Session()
                with sess.as_default():
                    #cnn = CNN( x_train.shape[1], y_train.shape[1], 
                    #        pretrained_embeddings, vocab_size, dim, filter_sizes, num_filters )
                    cnn = Hybrid_CNN( x_train.shape[1], y_train.shape[1], 
                            pretrained_embeddings, vocab_size, dim, filter_sizes, num_filters, md, md.shape[1], md_f_sizes, md_num_filters )
                    global_step = tf.Variable(0, name="global_step", trainable="False")
                    optimizer = tf.train.AdamOptimizer(0.001)
                    gradients = optimizer.compute_gradients( cnn.loss )
                    train_op = optimizer.apply_gradients( gradients, global_step=global_step )
                    

                    # Output directory for models and summaries
                    #timestamp = str(int(time.time()))
                    op_file = "d_"+ str(dim) + "_ep_" + str(num_epochs) + "_num_fil_" + str(num_filters) + "_fsize_" + "".join(str(x) for x in filter_sizes) + "_drp_prob_" + str( dropout_keep_prob )
                    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", op_file))
                    print("Writing to {}\n".format(out_dir))

                    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    
                    num_chkpts = 1
                    saver = tf.train.Saver(tf.global_variables(), num_chkpts)

                    # Write vocabulary
                    vocab_processor.save(os.path.join(out_dir, "vocab")) 

                    sess.run(tf.global_variables_initializer())

                    def train_step(x_batch, y_batch, m_batch, m_len_batch):
                        feed_dict = {
                          cnn.x_input: x_batch,
                          cnn.y_output: y_batch,
                          cnn.m_input: m_batch,
                          cnn.md_len: m_len_batch,
                          cnn.dropout_prob: 0.6
                        }
                        _, step, loss, accuracy = sess.run(
                            [train_op, global_step, cnn.loss, cnn.acc],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                    def dev_step(x_batch, y_batch, m_batch, m_len_batch ):
                        feed_dict = {
                          cnn.x_input: x_batch,
                          cnn.y_output: y_batch,
                          cnn.m_input: m_batch,
                          cnn.md_len: m_len_batch,
                          cnn.dropout_prob: 1.0
                        }
                        step, loss, accuracy = sess.run(
                            [global_step, cnn.loss, cnn.acc],
                            feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                    #Train the model in batches
                    data = list( zip( x_train, y_train, m_train, m_len_train))
                    data_batches  = gen_batch( data, batch_size, num_epochs )
                    for batch_data in data_batches:
                        x_batch, y_batch, m_batch, m_len_batch = zip( *batch_data)
                        train_step( x_batch, y_batch, m_batch, m_len_batch )
                        current_step = tf.train.global_step( sess, global_step )
                        if current_step % 20 == 0:
                            print("\nEvaluation:")
                            dev_step(x_dev, y_dev, m_dev, md_dev_len)
                            print("")
                        if current_step % 100 == 0:
                            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            print("Saved model checkpoint to {}\n".format(path))
                    
        #Testing step
        if test:
            rev_dict = { 0:'pants-fire', 1:'false', 2:'barely-true', 3:'half-true', 4:'mostly-true', 5:'true'}
            x_test, md_test, md_len_test = load_data_and_labels( test_file, test=True  )
            x_test = np.array(list(vocab_processor.fit_transform(x_test)))
            m_test = np.array(list(vocab_processor.fit_transform(md_test))) 
            chkpt_dir1 = out_dir + '/checkpoints'
            chkpt_file = tf.train.latest_checkpoint( chkpt_dir1 )

            graph = tf.Graph()
            with graph.as_default():
                sess_test = tf.Session()
                with sess_test.as_default():
                    saver = tf.train.import_meta_graph("{}.meta".format(chkpt_file))
                    saver.restore(sess_test, chkpt_file)

                    x_input = graph.get_operation_by_name("x_input").outputs[0]
                    m_input = graph.get_operation_by_name("m_input").outputs[0]
                    md_len = graph.get_operation_by_name("md_len").outputs[0]
                    dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]
                     
                    all_pred = []
                    
                    preds = graph.get_operation_by_name("fcl/pred").outputs[0]
                    #data  = list(zip(x_test_shuf, y_test_shuf))
                    #data_batches = gen_batch( data, batch_size, 1 )
                    data = list( zip( x_test, m_test, md_len_test ))
                    data_batches = gen_batch( data, batch_size, 1 )
                    
                    for batch_data in data_batches:
                        #x_batch_test, y_batch_test = zip( *batch_data)
                        x_batch_test, m_batch_test, m_len_batch_test = zip( *batch_data)
                        batch_pred = sess_test.run( preds, {x_input:x_batch_test, m_input: m_batch_test, md_len: m_len_batch_test, dropout_prob:1.0 })
                        all_pred = np.concatenate([all_pred, batch_pred])
            

            with open(output_file, 'w') as f:
                for p in all_pred:
                    f.write("%s\n" % rev_dict[p])


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()



def load_data_and_labels( in_file, test=False ):
    x_text = []                                                                                                                            
    labels = []
    raw_labels = []
    dict_label = {'pants-fire': 0, 'false':1, 'barely-true':2, 'half-true':3, 'mostly-true':4, 'true':5} 
    metadata = []
    md_len = []
    s = [0,1,2,3,4,5]
    num_labels = 6 
    m = np.zeros(( len(s), num_labels ))
    m[ np.arange(num_labels), s ] = 1 
    m = m.tolist()
    
    with open( in_file, "r" ) as f:
            content = f.readlines()
        
    for line in content:
        split_line = line.split('\t')
        if not test: 
            label, stmt, sub, speaker, title, state_info, party, context = split_line
            x_text.append( clean_str( stmt.strip() ) ) 
            labels.append( m[dict_label[label]] )
            raw_labels.append( dict_label[label] )
            #metadata.append( [sub,speaker,title,state_info,party+context] )
            to_be_app = clean_str(sub) + " " + clean_str(speaker) + " "+ clean_str(title) + " " + clean_str(state_info) + " " + clean_str(party) + " " + clean_str(context)
            metadata.append( to_be_app ) 
            md_len.append( len(to_be_app.split()) )
        else:
            stmt, sub, speaker, title, state_info, party, context = split_line
            x_text.append( clean_str( stmt.strip() ) ) 
            #metadata.append( [sub,speaker,title,state_info,party+context] )
            to_be_app = clean_str(sub) + " " + clean_str(speaker) + " "+ clean_str(title) + " " + clean_str(state_info) + " " + clean_str(party) + " " + clean_str(context)
            metadata.append( to_be_app ) 
            md_len.append( len(to_be_app.split()) )
            
    if test:
        return x_text, metadata, md_len
    else:
        return x_text, labels, raw_labels, metadata, md_len


def process_metadata( metadata, index = None ):
    md = []
    max_len = 0
    if not index:
        for m in metadata:
            cnt = 0
            l = []
            for c in m:
                for w in c.split():
                    for clean_w in clean_str(w).split():
                        l.append( clean_w )
                        cnt += 1
            md.append( l )
            if cnt > max_len:
                max_len = cnt

    return md, max_len
                


def build_vocab( x_text, x_dev, x_test,  metadata=None ):
    max_document_length = max([len(x.split(" ")) for x in x_text])
    max_len_md = max([len(x.split(" ")) for x in metadata]) 
    #print("Max Len MD:", max_len_md)
    #for x in x_text:
    #    print (len(x.split(" ")))
    #print ("Meidan:", statistics.median([len(x.split(" ")) for x in x_text]))
    #print ("Max:", max_document_length)
    #max_document_length = 35
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text + metadata + x_dev + x_test )))
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    return x, vocab_processor

def shuffle_data( x, y, md, md_len ):
    np.random.seed(10)
    y = np.asarray(y)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    print (shuffle_indices)
    #x_shuffled = x[shuffle_indices]
    #y_shuffled = y[shuffle_indices]
    #m_shuffled = md[shuffle_indices]
    #m_len_shuffled = md_len[shuffle_indices]
    x_shuffled, y_shuffled, m_shuffled, m_len_shuffled = x, y, md, md_len
    return x_shuffled, y_shuffled, m_shuffled, m_len_shuffled

def split_data( x_shuffled, y_shuffled, y ):
    train_prop = 0.90
    train_idx = int( train_prop * float(len(y)))
    x_train, x_dev = x_shuffled[:train_idx], x_shuffled[train_idx:]
    y_train, y_dev = y_shuffled[:train_idx], y_shuffled[train_idx:]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, x_dev, y_train, y_dev

def load_embed_dict( embed_file ):
    ret = OrderedDict()
    for row in list(open(embed_file,"r").readlines()):
        fields = row.strip().split()
        word = fields[0]
        ret[word] = np.array(list(map(float, fields[1:])))

    return ret

def load_embeddings(embed_file, vocab_processor, dim ):
    vocab_dict = vocab_processor.vocabulary_._mapping
    embeddings_cache_file = embed_file + ".cache.npy"

    embeddings = np.array(np.random.randn(len(vocab_dict)+ 1, dim), dtype=np.float32)
    
    embeddings[0] = np.array(np.random.uniform(-1, 1, size=(1,dim)))
    for word, vec in load_embed_dict(embed_file).items():
        if word in vocab_dict:
            embeddings[vocab_dict[word]] = vec
   
    np.save(embeddings_cache_file, embeddings)
    print("Initialized embeddings")
    return embeddings

def gen_batch( data, batch_size, epochs ):
    data = np.array( data )
    size = len( data )

    num_batches = int( (size - 1)/batch_size) + 1
    for e in range( epochs ):
        for batch in range( num_batches ):
            start = batch * batch_size
            end = min( (batch + 1) * batch_size, size )  
            yield data[start:end]

if __name__ == '__main__':
    run(train_file, valid_file , test_file, output_file)

