#!/usr/bin/env python
# coding: utf-8

# In[11]:


#!/usr/bin/env python
# coding: utf-8
# Two hidde layer network
# In[1]:


import numpy as np
import tensorflow as tf
import etp as etp
import calculate_virtual_coordinates as cvc
import matplotlib.pyplot as plt

from random import seed
from random import randint

import pickle
seed(23)

C = 5 # maximum number of anchors
H1 = 500 # size of hidden filters
H2 = 500 # size of hidden filters
F = 2 # Final dimension of coordinates
random_seed = 23
radius = 1 # extent of possible communication
MAX_NODES = 1000
learning_rate = 0.2
epochs = 1000
display_cost_period = 10
num_iter = 5


# In[2]:


def plot_learning(cost_history):
    plt.plot(cost_history)
    plt.xlabel('Number of Graphs')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.show()


# In[3]:


g = tf.Graph()

with g.as_default():
    tf.set_random_seed(random_seed)
    W_0 = tf.Variable(tf.truncated_normal(shape=(C, H1),
                                      mean=0.0,
                                      stddev=0.1,
                                      dtype=tf.float32,
                                      seed=random_seed), name='wt_1')
    
    W_1 = tf.Variable(tf.truncated_normal(shape=(H1, H2),
                                  mean=0.0,
                                  stddev=0.1,
                                  dtype=tf.float32,
                                  seed=random_seed), name='wt_2')

    W_2 = tf.Variable(tf.truncated_normal(shape=(H2, F),
                                    mean=0.0,
                                    stddev=0.1,
                                    dtype=tf.float32,
                                    seed=random_seed), name='wt_3')
    

    input_layer = tf.placeholder(tf.float32, [None, C], # INPUT: num_of_nodes X VC_values
                                 name='input')
    
    physical_coordinates = tf.placeholder(tf.float32, [None, F], # INPUT: num_of_nodes X geographic_coordinates
                                 name='phy_coord')
    
    adj = tf.placeholder(tf.float32, [None, None], # INPUT: num_of_nodes X num_of_nodes
                                 name='adj')
    
    A_caret = tf.placeholder(tf.float32, [None, None], # INPUT: num_of_nodes X num_of_nodes
                                 name='A_caret')

    
    out_1 = tf.matmul(tf.matmul(A_caret, input_layer), W_0)
    out_1 = tf.nn.relu(out_1)
    out_2 = tf.matmul(tf.matmul(A_caret, out_1), W_1)
    out_2 = tf.nn.relu(out_2)
    
    out_3 = tf.matmul(tf.matmul(A_caret, out_2), W_2)
    
    
    cost = tf.losses.mean_squared_error(out_3, physical_coordinates)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    
    saver = tf.train.Saver()
    

    

 


# In[ ]:



# In[12]:


cost_history = []
etp_history = []
with tf.Session(graph=g) as sess:
    # `sess.graph` provides access to the graph used in a `tf.Session`.
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    #saver.restore(sess, 'model_h_2_g_1000_alpha_7.ckpt')

    # Perform your computation...
    for i in range(epochs):
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        phy_coord, inps = cvc.get_VC(randint(10, MAX_NODES))
        adj_matrix = cvc.get_adj(phy_coord)
        A_caret_matrix = cvc.normalize_adj(adj_matrix + np.eye(adj_matrix.shape[0]))
        
        output_1 = sess.run(out_3, feed_dict={input_layer: inps,
                                            physical_coordinates: phy_coord,
                                            adj: adj_matrix,
                                            A_caret: A_caret_matrix})
        #print('output of untrained neural network : {}'.format(output_1))

        print('Epoch: {}\t ETP-1(untrained neural network) : {}'.format(i, etp.get_best_etp(output_1, phy_coord)))

        
               
        
    


        
        
        
        
            

    writer.close()




with open('h_2', 'wb') as fp:
    pickle.dump(cost_history, fp)








