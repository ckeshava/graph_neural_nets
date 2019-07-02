import numpy as np
import os

import datetime
from TPM_from_VCS.util import tpm_from_vcs as get_TPM
from TPM_from_VCS.util import broadcast_vector as BV
from TPM_from_VCS import config
from TPM_from_VCS.util import calculate_virtual_coordinates as calc_VCS


# returns the physical coordinates and virtual coordinates.
def create_dataset(size):
    # dataset variable must be initialised outside the for-loop. Otherwise, only 1 training-datapoint is prepared!
    dataset = []
        
    # for debug only:
 
    for i in range(0, size):
        temp = np.random.randint(low=6, high=100)
        # print('Number of nodes currently : {}'.format(temp))

        phy_coordinates, vcs = calc_VCS.get_VC(temp) # 20 sensor nodes in the network

        for i in range(0, phy_coordinates.shape[0]):
            dataset.append(((vcs[i]), (phy_coordinates[i])))


        # if i % 50 == 0:
            # print('Created {} records in the dataset\n'.format(i))

        
    
    return dataset

def save_dataset(dataset, filename):
    np.save(os.path.join(config.dataset_dir, filename + '.npy'), dataset)
