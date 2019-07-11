
# coding: utf-8

# In[5]:


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
MIN_NODES = 200
learning_rate = 0.2
num_of_graphs = 100
display_cost_period = 10
num_iter = 5


# In[2]:


def plot_learning(cost_history):
    plt.plot(cost_history)
    plt.xlabel('Number of Graphs')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.show()



# In[7]:



data = {}
for i in range(num_of_graphs):
    phy_coord, inps = cvc.get_VC(randint(MIN_NODES, MAX_NODES))
    adj_matrix = cvc.get_adj(phy_coord)
    A_caret_matrix = cvc.normalize_adj(adj_matrix + np.eye(adj_matrix.shape[0]))
    
    data[i] = {'VC': inps, 'PC': phy_coord, 'Adj': adj_matrix, 'Lap': A_caret_matrix}
    print('Epoch:{}'.format(i))
        


# In[9]:


with open('max_1000_min_200_graphs_100_area_200.pickle', 'wb') as fp:
    pickle.dump(data, fp)

