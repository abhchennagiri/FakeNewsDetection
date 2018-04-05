import tensorflow as tf
import numpy as np

class CNN( object ):
    """
    Simple CNN model without using any metadata information
    """
    def __init__( self, seq_length, num_labels, pretrained_embeddings, vocab_size, dim, filter_sizes, num_filters ):
        self.x_input = tf.placeholder( tf.int32, [None, seq_length], name="x_input" )
        self.y_output = tf.placeholder( tf.int32, [None, num_labels], name="y_output" )
        self.dropout_prob = tf.placeholder( tf.float32, name="dropout_prob" )

        #Embedding layer
        init_embeddings = tf.Variable( pretrained_embeddings )
        embed_size = pretrained_embeddings.shape[1]
        self.x_embed = tf.nn.embedding_lookup( init_embeddings, self.x_input )
        self.x_embed_4d = tf.expand_dims(self.x_embed, -1)

        #Convolution layer 
        p_op = []
        padding = "VALID"
        strides = [1,1,1,1]
        stddev = 0.1
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

        #Predictions
        with tf.name_scope("fcl"):
            W = tf.get_variable( "W", shape=[ total_filters, num_labels ], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable( tf.constant( 0.1, shape = [num_labels] ), name = "b" )
            
            self.softmax_op = tf.nn.xw_plus_b( self.h_pool_drop, W, b, name="softmax_op" )
            self.pred = tf.argmax( self.softmax_op, 1, name="pred" )

        #Mean Cross Entropy Loss
        with tf.name_scope("MCE"):
            loss = tf.nn.softmax_cross_entropy_with_logits( logits=self.softmax_op, labels=self.y_output )
            self.loss = tf.reduce_mean( loss ) 

        #Accuracy scores
        with tf.name_scope("accuracy"):
            true_pred = tf.equal( self.pred, tf.argmax( self.y_output, 1 ) )
            self.acc  = tf.reduce_mean( tf.cast( true_pred, "float" ), name="acc" )
