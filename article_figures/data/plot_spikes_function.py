

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
import sys 

## Functions to plot the spikes trains of MFs, GrCs and GoCs

def plot_spikes(oi, ot, dict, t, dt):

    ## Receives cells spikes times and indexes, a dictionary with important parameters about the simulation
    ## and initial time to plot and the interval to plot
    
    ## Returns a figure plot
    
    ## Reading parameters
    s = dict["seed"]
    num_patterns = dict["num_patterns"]
    duration_pattern = dict["duration_pattern"]
  
    ## Loading information about the structure of the model
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)
        ## Number of cells for each population
    n_inputs = glos.shape[0]    
    with open('../../data/structure/seed' + str(s) + '/grcs.npy', 'rb') as f:
        grcs = np.load(f)
    n_hidden_neurons = grcs.shape[0]

    ## boolean vector to show or not each pattern (by default, showing all)
    show_pattern = np.arange(0, num_patterns, 1)
    ## computing vectors with pattern durations 
    input_duration_patterns = duration_pattern*np.ones((num_patterns))
    cumsum_input_duration_patterns = np.cumsum(input_duration_patterns)
    cumsum_input_duration_patterns = np.insert(cumsum_input_duration_patterns, 0, 0)
    cumsum_input_duration_patterns = cumsum_input_duration_patterns[:num_patterns]
    show_pattern = show_pattern[:num_patterns]
    
    plt.rcParams['figure.figsize'] = (10,3)
    plt.rcParams["savefig.facecolor"] = "white"
    ## Creating figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [10, 20, 6]})

    fill_pattern = np.roll(np.repeat(show_pattern, 2), 1)

    ## For mossy fibers
    vector_glos = np.ones((oi.shape[0]), dtype = bool)

    i = 0
    for indice in oi: 
        if indice in range(n_inputs): 
            vector_glos[i] = True
        else:
            vector_glos[i] = False
        i += 1   

    ## Colouring the patterns
    for i in range(1, num_patterns+1):

        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        
        ax1.fill_between(
            np.repeat(cumsum_input_duration_patterns, 2),
            0,
            1.0,
            where=fill_pattern==i,
            facecolor='C{}'.format(i-1), alpha=0.5, transform=trans)
        
        ax1.axvline(x = i*duration_pattern, color = 'black')
        ax2.axvline(x = i*duration_pattern, color = 'black')
        ax3.axvline(x = i*duration_pattern, color = 'black')

    ax1.set(ylabel='MF')

    ## Scattering the spikes of MFs
    tim = (ot>=t) * (ot<=(t+dt))
    if dt<=10.0:
        inp = (oi<= (n_inputs - 1)) * vector_glos * (oi%10 == 0)
        ax1.scatter(ot[inp*tim], oi[inp*tim], s=4.0, color='red')

    ## For GrCs
    ## Scattering the spikes of GrCs
    vector_grcs = np.ones((oi.shape[0]), dtype = bool)
    i = 0
    for indice in oi: 
        if indice in range(n_inputs, n_hidden_neurons + n_inputs, 1): 
            vector_grcs[i] = True
        else:
            vector_grcs[i] = False
        i += 1   

    out = (oi > (n_inputs - 1))*(oi <= (n_inputs + n_hidden_neurons - 1)) * vector_grcs * (oi%50 == 0)
    ax2.scatter(ot[out*tim], oi[out*tim], s=4.0, color='darkblue')
    ax2.axes.get_yaxis().set_visible(True)   
    ax2.set(ylabel='GRC')

    ## For GoCs
    ## Scaterring the spikes of GoCs
    out = (oi > (n_inputs + n_hidden_neurons - 1))
    ax3.scatter(ot[out*tim], oi[out*tim], s=4.0, color='green')
    ax3.axes.get_yaxis().set_visible(True)
    ax3.set(ylabel='GOC')
    ax3.set(xlabel = 'Time (s)')
    ax3.set_yticklabels([0, 0, 5])

    plt.xlim((t, t+dt))

    return fig


## Function to illustrate a specific example for the project article
def plot_spikes_improved(oi, ot, dict, t, dt):

    s = dict["seed"]
    num_patterns = dict["num_patterns"]
    duration_pattern = dict["duration_pattern"]
  
    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)
        ## Number of cells for each population
    n_inputs = glos.shape[0]    

    with open('../../data/structure/seed' + str(s) + '/grcs.npy', 'rb') as f:
        grcs = np.load(f)

    n_hidden_neurons = grcs.shape[0]

    plt.rcParams['figure.figsize'] = (10,3)
    plt.rcParams["savefig.facecolor"] = "white"
    show_pattern = np.arange(0, num_patterns, 1)
    input_duration_patterns = duration_pattern*np.ones((num_patterns))
    cumsum_input_duration_patterns = np.cumsum(input_duration_patterns)
    cumsum_input_duration_patterns = np.insert(cumsum_input_duration_patterns, 0, 0)
    cumsum_input_duration_patterns = cumsum_input_duration_patterns[:num_patterns]
    show_pattern = show_pattern[:num_patterns]


    plt.rcParams['figure.figsize'] = (10,4)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [10, 20, 6]})

    x = np.append(show_pattern, 10)
    fill_pattern = np.roll(np.repeat(x, 2), 1)

    vector_glos = np.ones((oi.shape[0]), dtype = bool)

    i = 0
    for indice in oi: 
        if indice in range(n_inputs): 
            vector_glos[i] = True
        else:
            vector_glos[i] = False
        i += 1   


    colors = ['hotpink', 'deepskyblue', 'green', 'hotpink', 'orange', 'green', 'purple', 'deepskyblue', 'hotpink', 'brown', 'maroon']
    for i in range(1, num_patterns+1):


        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        trans2 = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
        trans3 =  mtransforms.blended_transform_factory(ax3.transData, ax3.transAxes)
        
        labels = [1, 4, 7, 1, 3, 7, 9, 4, 1, 6, 8, 9]
        result = np.append(cumsum_input_duration_patterns, cumsum_input_duration_patterns[-1] + duration_pattern)

        ax1.fill_between(
            np.repeat(result, 2),
            0,
            1.0,
            where=fill_pattern==i-1,
            facecolor=colors[i-1], alpha=0.5, transform=trans)
        
                    
        ax1.text(-0.05 + i*0.08, 340.0, 'P{}'.format(labels[i-1]), c=colors[i-1], fontsize = 16)
        ax1.axvline(x = i*duration_pattern, color = 'black')
        ax2.axvline(x = i*duration_pattern, color = 'black')
        ax3.axvline(x = i*duration_pattern, color = 'black')

    ax1.set_ylabel('MF', fontsize = 15, color = 'red')

    tim = (ot>=t) * (ot<=(t+dt))


    if dt<=10.0:
        inp = (oi<=n_inputs) * vector_glos * (oi%10 == 0)
        ax1.scatter(ot[inp*tim], oi[inp*tim], s=4.0, color='red')


    vector_grcs = np.ones((oi.shape[0]), dtype = bool)
    i = 0
    for indice in oi: 
        if indice in range(n_inputs, n_inputs + n_hidden_neurons): 
            vector_grcs[i] = True
        else:
            vector_grcs[i] = False
        i += 1   

    out = (oi > (n_inputs - 1))*(oi <= (n_inputs + n_hidden_neurons - 1)) * vector_grcs * (oi%50 == 0)
    ax2.scatter(ot[out*tim], oi[out*tim], s=4.0, color='darkblue')
    ax2.axes.get_yaxis().set_visible(True)   
    ax2.set_ylabel('GrC', fontsize = 15, color = 'darkblue')
    ticksy = [0, 1500, 3000]
    ax2.set_yticks(ticksy)
    ax2.set_yticklabels(ticksy)


    out = (oi > (n_inputs + n_hidden_neurons - 1))
    #print(out)
    ax3.scatter(ot[out*tim], oi[out*tim], s=4.0, color='green')
    ax3.axes.get_yaxis().set_visible(True)
    ax3.set_ylabel('GoC', fontsize = 15, color = 'green')
    ax3.set_xlabel('Time (s)', fontsize = 15)
    #ax3.set_yticks([0, 5])
    ax3.set_yticklabels([0, 0, 5])
    ax1.tick_params('both', labelsize = 14)
    ax2.tick_params('both', labelsize = 14)
    ax3.tick_params('both', labelsize = 14)
    plt.xlim((t, t+dt))
        
    return fig