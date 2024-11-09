

import numpy as np 
import sys

## Functions that generate the input patterns configuration for the simulation

def mf_activity_overview(dict):

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

def mf_activity_1(dict):

    ## Receives a parameter dictionary
    ## Returns the MF activity for the simulation of the first stimulus (auditory stimulus)
    
    ## Reading parameters
    f_mf = np.linspace(0.05, 0.95, 10)
    s = dict["seed"]
    sigma = dict["sigma"]
    fraction = f_mf[dict["fraction"]]
    num_patterns = dict["num_patterns"]
    pattern = dict["pattern"]

    ## Loading MF structure and activity
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)

    n_inputs = glos.shape[0]

    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction) + '_s_' + str(sigma) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
        
    mf_activity = mf_activity[0].T

    ## Creating vector for noise and for stimulus
    mf_activity_2 = np.zeros((n_inputs, 1))
    mf_activity_1 = np.zeros((n_inputs, 1))

    ## Introducing initial noise before stimulus
    mf_activity_1[:,0] = np.random.choice([0, 1], size=(n_inputs,), p=[14./15, 1./15])

    ## Retrieving selected pattern (stimulus)
    for i in range(n_inputs):
        mf_activity_2[i, 0] = mf_activity[i, pattern]

    ## Noise 1/3 of simulation time and stimulus 2/3 of simulation time
    mf_activity_1 = np.repeat(mf_activity_1, int((num_patterns)/3), axis = 1)
    mf_activity_2 = np.repeat(mf_activity_2, int((num_patterns*2)/3), axis = 1)
    mf_activity = np.concatenate((mf_activity_1, mf_activity_2), axis = 1)

    return mf_activity

def mf_activity_2(dict):

    ## Receives a parameter dictionary
    ## Returns the MF activity for the simulation of the second stimulus (somatosensory stimulus)
        
    ## Reading parameters
    f_mf = np.linspace(0.05, 0.95, 10)
    seed = s = dict["seed"]
    sigma = dict["sigma"]
    fraction = f_mf[dict["fraction"]]
    num_patterns = dict["num_patterns"]
    pattern = dict["pattern"]

    ## Loading MF structure and activity
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)
        ## Number of cells for each population
    n_inputs = glos.shape[0]

    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction) + '_s_' + str(sigma) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
        
    mf_activity = mf_activity[0].T

    ## Creating two vectors for noise and for stimulus
    mf_activity_1 = np.zeros((n_inputs, 1))
    mf_activity_2 = np.zeros((n_inputs, 1))
    mf_activity_3 = np.zeros((n_inputs, 1))

    ## Introducing initial noise before and after stimulus
    mf_activity_1[:,0] = np.random.choice([0, 1], size=(300,), p=[99./100, 1./100])
    mf_activity_3[:,0] = np.random.choice([0, 1], size=(300,), p=[19./20, 1./20])

    ## Retrieving selected pattern (stimulus)    
    for i in range(n_inputs):
        mf_activity_2[i, 0] = mf_activity[i, pattern]

    ## Noise before and after stimulus   
    mf_activity_1 = np.repeat(mf_activity_1, 4, axis = 1)
    mf_activity_2 = np.repeat(mf_activity_2, 2, axis = 1)
    mf_activity_3 = np.repeat(mf_activity_3, 6,axis = 1)

    mf_activity = np.concatenate((mf_activity_1, mf_activity_2, mf_activity_3), axis = 1)

    return mf_activity
