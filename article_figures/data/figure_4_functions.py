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
    ## Returns the MF activity for the simulation of one stimulus
    
    ## Reading parameters
    f_mf = np.linspace(0.05, 0.95, 10)
    s = dict["seed"]
    sigma = dict["sigma"]
    fraction = f_mf[dict["fraction"]]
    num_patterns = dict["num_patterns"]
    pattern = dict["pattern"]

    # Loading structure and MF activity
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)

    n_inputs = glos.shape[0]

    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction) + '_s_' + str(sigma) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
        
    mf_activity = mf_activity[0].T
    
    ## Creating vector for stimulus
    mf_activity_2 = np.zeros((n_inputs, 1))

    ## Retrieving selected pattern (stimulus)
    for i in range(n_inputs):
        mf_activity_2[i, 0] = mf_activity[i, pattern]

    ## Repeating stimulus during the whole simulation
    mf_activity = np.repeat(mf_activity_2, num_patterns, axis = 1)

    return mf_activity

def mf_activity_combined(dict1, dict2):

    ## Receives a parameter dictionary
    ## Returns the MF activity for the simulation of two stimuli combined

    ## Reading parameters
    f_mf = np.linspace(0.05, 0.95, 10)
    s = dict1["seed"]
    sigma1 = dict1["sigma"]
    sigma2 = dict2["sigma"]
    fraction1 = f_mf[dict1["fraction"]]
    fraction2 = f_mf[dict2["fraction"]]
    num_patterns = dict1["num_patterns"]
    pattern1 = dict1["pattern"]
    pattern2 = dict2["pattern"]

    ## Loading structure and MF activity for each combination of parameters
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)

    n_inputs = glos.shape[0]

    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction1) + '_s_' + str(sigma1) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
        
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction2) + '_s_' + str(sigma2) + '.npy', 'rb') as f: 
        mf_activity1 = np.load(f)
        
    mf_activity = mf_activity[0].T 
    mf_activity1 = mf_activity1[0].T

    ## Creating vector to store the combination of both stimuli
    mf_activity_2 = np.zeros((n_inputs, 1))

    ## For inactive MFs in one stimulus, we check if they are active in the second one
    ## If they are, we activate them
    for i in range(n_inputs):
        mf_activity_2[i, 0] = mf_activity[i, pattern1]
        if mf_activity_2[i,0] == 0:
            mf_activity_2[i, 0] = mf_activity1[i, pattern2]

    mf_activity = np.repeat(mf_activity_2, num_patterns, axis = 1)

    return mf_activity


def mf_activity_trials(dict):

    ## Receives a parameter dictionary
    ## Returns the MF activity for the simulation of one stimulus with initial noise

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

    n_inputs = glos.shape[0]


    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction) + '_s_' + str(sigma) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
        
        
    mf_activity = mf_activity[0].T

    ## Creating vector for noise and for stimulus
    mf_activity_2 = np.zeros((n_inputs, 1))
    mf_activity_1 = np.zeros((n_inputs, 1))

    ## Introducing initial noise before stimulus
    mf_activity_1[:,0] = np.random.choice([0, 1], size=(n_inputs,), p=[7./8, 1./8])

    ## Retrieving selected pattern (stimulus)
    for i in range(n_inputs):
        mf_activity_2[i, 0] = mf_activity[i, pattern]

    ## Noise 1/3 of simulation time and stimulus 2/3 of simulation time
    mf_activity_1 = np.repeat(mf_activity_1, int((num_patterns)/3), axis = 1)
    mf_activity_2 = np.repeat(mf_activity_2, int((num_patterns*2)/3), axis = 1)
    mf_activity = np.concatenate((mf_activity_1, mf_activity_2), axis = 1)

    return mf_activity



def mf_activity_combined_trials(dict1, dict2):

    ## Receives a parameter dictionary
    ## Returns the MF activity for the simulation of two stimuli combined, with initial noise

    f_mf = np.linspace(0.05, 0.95, 10)
    s = dict1["seed"]
    sigma1 = dict1["sigma"]
    sigma2 = dict2["sigma"]
    fraction1 = f_mf[dict1["fraction"]]
    fraction2 = f_mf[dict2["fraction"]]
    num_patterns = dict1["num_patterns"]
    pattern1 = dict1["pattern"]
    pattern2 = dict2["pattern"]

    ## Loading structure and MF activity for for each combination of parameters
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)
    n_inputs = glos.shape[0]

    sys.path.insert(1, '../../data/input_patterns')
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction1) + '_s_' + str(sigma1) + '.npy', 'rb') as f:       
        mf_activity = np.load(f)
        
    with open('../../data/input_patterns/seed'+ str(s) + '/mf_activity_f_mf_' + str(fraction2) + '_s_' + str(sigma2) + '.npy', 'rb') as f: 
        mf_activity1 = np.load(f)
        
    
    mf_activity = mf_activity[0].T 
    mf_activity1 = mf_activity1[0].T

     ## Creating vector to store the combination of both stimuli
    mf_activity_2 = np.zeros((n_inputs, 1))

    ## For inactive MFs in one stimulus, we check if they are active in the second one
    ## If they are, we activate them
    for i in range(n_inputs):
        mf_activity_2[i, 0] = mf_activity[i, pattern1]
        if mf_activity_2[i,0] == 0:
            mf_activity_2[i, 0] = mf_activity1[i, pattern2]

    ## Introducing noise before the stimuli combination
    mf_activity_1 = np.zeros((n_inputs, 1))
    mf_activity_1[:,0] = np.random.choice([0, 1], size=(300,), p=[7./8, 1./8])

    ## Noise 1/3 of simulation time and stimuli combined 2/3 of simulation time
    mf_activity_1 = np.repeat(mf_activity_1, int((num_patterns)/3), axis = 1)
    mf_activity_2 = np.repeat(mf_activity_2, int((num_patterns)*2/3), axis = 1)
    mf_activity = np.concatenate((mf_activity_1, mf_activity_2), axis = 1)

    return mf_activity