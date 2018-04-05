import tensorflow as tf
import numpy as np

class Hybrid_CNN( object ):
    """
    Hybrid CNN model without using metadata information
    """
    def __init__( self, seq_length, num_labels, pretrained_embeddings, vocab_size, dim, filter_sizes, num_filters, metadata, md_seq_len,  md_f_sizes, md_num_filters ):
        self.x_input = tf.placeholder( tf.int32, [None, seq_length], name="x_input" )
        self.y_output = tf.placeholder( tf.int32, [None, num_labels], name="y_output" )
        self.m_input = tf.placeholder( tf.int32, [None, seq_length], name="m_input"  )
        self.dropout_prob = tf.placeholder( tf.float32, name="dropout_prob" )
        self.md_len = tf.placeholder( tf.int32, [None], name="md_len" )

        #Embedding layer
        init_embeddings = tf.Variable( pretrained_embeddings )
        embed_size = pretrained_embeddings.shape[1]
        self.x_embed = tf.nn.embedding_lookup( init_embeddings, self.x_input )
        self.m_embed = tf.nn.embedding_lookup( init_embeddings, self.m_input )
        self.m_embed_4d = tf.nn.embedding_lookup( self.m_embed, -1 )
        self.x_embed_4d = tf.expand_dims(self.x_embed, -1)

        #Convolution layer 
        p_op = []
        strides = [1,1,1,1]
        stddev = 0.1
        padding = "VALID"

        for f_size in filter_sizes:
            with tf.name_scope("conv-filter%s"% f_size):
                filter_shape = [ f_size, dim, 1, num_filters ]
                # Below could be modified
                W = tf.Variable(tf.truncated_normal( filter_shape, stddev=stddev ), name="W")
                b = tf.Variable( initial_value = tf.constant(0.1, shape=[num_filters]), name="b" )
                #b = tf.Variable( initial_value = tf.zeros([num_filters]), name="b" )

                conv_layer = tf.nn.conv2d(self.x_embed_4d, W, strides = strides, padding=padding, name="conv_layer")

                #Applying Relu
                h = tf.nn.relu(tf.nn.bias_add(conv_layer, b), name = "h")

                #Max-pooling
                max_pool = tf.nn.max_pool(h, ksize=[1, seq_length-f_size + 1, 1, 1], strides=strides, padding=padding, name="max_pool")

                p_op.append( max_pool )

        total_filters = len(filter_sizes) * num_filters
        self.h_pool = tf.concat( p_op, 3 )
        self.h_pool_flat = tf.reshape( self.h_pool, [-1, total_filters] )

        #Add dropout here
        self.h_pool_drop = tf.nn.dropout( self.h_pool_flat, self.dropout_prob )

        with tf.name_scope("biLSTM"):
            state_len = dim

            cell = tf.nn.rnn_cell.LSTMCell(num_units=state_len, state_is_tuple=True)
            outputs, states  = tf.nn.bidirectional_dynamic_rnn( cell_fw=cell, cell_bw=cell, dtype=tf.float32, 
                                                                sequence_length=self.md_len, inputs = self.m_embed )

            fw_states, bw_states = states
            fw_op, bw_op = outputs

            encoded = tf.stack([fw_op, bw_op], axis=3)
            md_op = []

            for f_size in md_f_sizes:
                with tf.name_scope("hyb-filter%s"% f_size):
                    filter_shape = [ f_size, dim, 2, md_num_filters ]
                    # Below could be modified
                    W = tf.Variable(tf.truncated_normal( filter_shape, stddev=stddev ), name="W")
                    b = tf.Variable( initial_value = tf.constant(0.1, shape=[md_num_filters]), name="b" )
                    #b = tf.Variable( initial_value = tf.zeros([num_filters]), name="b" )

                    conv_layer = tf.nn.conv2d(encoded, W, strides = strides, padding=padding, name="conv_layer")

                    #Applying Relu
                    h = tf.nn.relu(tf.nn.bias_add(conv_layer, b), name = "h")

                    #Max-pooling
                    max_pool = tf.nn.max_pool(h, ksize=[1, seq_length-f_size + 1, 1, 1], strides=strides, padding=padding, name="max_pool")

                    md_op.append( max_pool )

            total_filters_md = len(md_f_sizes) * md_num_filters
            self.h_pool_md = tf.concat( md_op, 3 )
            self.h_pool_flat_md = tf.reshape( self.h_pool_md, [-1, total_filters_md] )
            
            #Add dropout here
            self.h_pool_md_drop = tf.nn.dropout( self.h_pool_flat_md, self.dropout_prob )

        print (self.h_pool_drop, self.h_pool_md_drop )    
        self.h_pred = tf.concat( (self.h_pool_drop, self.h_pool_md_drop), axis=1)

        #Predictions
        with tf.name_scope("fcl"):
            W = tf.get_variable( "W", shape=[ total_filters+total_filters_md, num_labels ], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable( tf.constant( 0.1, shape = [num_labels] ), name = "b" )
            
            self.softmax_op = tf.nn.xw_plus_b( self.h_pred, W, b, name="softmax_op" )
            self.pred = tf.argmax( self.softmax_op, 1, name="pred" )

        #Mean Cross Entropy Loss
        with tf.name_scope("MCE"):
            loss = tf.nn.softmax_cross_entropy_with_logits( logits=self.softmax_op, labels=self.y_output )
            self.loss = tf.reduce_mean( loss ) 

        #Accuracy scores
        with tf.name_scope("accuracy"):
            true_pred = tf.equal( self.pred, tf.argmax( self.y_output, 1 ) )
            self.acc  = tf.reduce_mean( tf.cast( true_pred, "float" ), name="acc" )
