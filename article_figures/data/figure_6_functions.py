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
 
    ## Loading activity
    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction) + '_s_' + str(sigma) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
    ## preparing activity dataset 
    mf_activity = mf_activity[0].T
    mf_activity = mf_activity[:,:num_patterns] ## taking only the number of patterns we want to simulate

    return mf_activity