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

    ## Loading structure and MF activity
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)
        ## Number of cells for each population
    n_inputs = glos.shape[0]

    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction) + '_s_' + str(sigma) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
        
    mf_activity = mf_activity[0].T

    ## Creating vector for noise and for stimulus
    mf_activity_2 = np.zeros((n_inputs, 1))
    mf_activity_1 = np.zeros((n_inputs, 1))

    ## Introducing initial noise before stimulus
    mf_activity_1[:,0] = np.random.choice([0, 1], size=(n_inputs,), p=[19./20, 1./20])

    ## Retrieving selected pattern (stimulus)
    for i in range(n_inputs):
        mf_activity_2[i, 0] = mf_activity[i, pattern]

    ## Introducing noise during the initial 2/7 time of the simulation before the stimulus
    mf_activity_1 = np.repeat(mf_activity_1, 2, axis = 1)
    mf_activity_2 = np.repeat(mf_activity_2, num_patterns-2, axis = 1)    
    mf_activity = np.concatenate((mf_activity_1, mf_activity_2), axis = 1)

    return mf_activity