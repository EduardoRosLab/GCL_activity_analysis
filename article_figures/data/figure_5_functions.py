

import matplotlib.pyplot as plt
import numpy as np
import sys

def mf_activity(dict):

    ## Receives a parameter dictionary
    ## Returns the MF activity for the simulation

    ## Reading parameters
    f_mf = np.linspace(0.05, 0.95, 10)
    s = dict["seed"]
    sigma = dict["sigma"]
    fraction = f_mf[dict["fraction"]]
    num_patterns = dict["num_patterns"]
    pattern = dict["pattern"]

    ## Loading MF activit
    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction) + '_s_' + str(sigma) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)

    ## Preparing activity dataset 
    mf_activity = mf_activity[0].T
    mf_activity = mf_activity[:,:num_patterns] ## taking only the number of patterns we want to simulate

    ## Combining and repeating some patterns 
    mf_activity[:,1] = mf_activity[:,1]
    mf_activity[:,2] = mf_activity[:,2]
    mf_activity[:,3] = mf_activity[:,0]
    mf_activity[:,4] = mf_activity[:,4]
    mf_activity[:,5] = mf_activity[:,2]
    mf_activity[:,6] = mf_activity[:,6]
    mf_activity[:,7] = mf_activity[:,1]
    mf_activity[:,8] = mf_activity[:,0]
    mf_activity[:,9] = mf_activity[:,9]

    return mf_activity
    




