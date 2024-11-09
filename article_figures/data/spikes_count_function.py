
import numpy as np 

def spikes_count(n_inputs, n_hidden_neurons, n_outputs, oi, ot, t_step, t_lower, total_simulation_time):

    total_rows = n_inputs + n_hidden_neurons + n_outputs # numero de neuronas
    total_columns = int(total_simulation_time / t_step) # numero de intervalos 
    spikes_count = np.zeros((total_rows, total_columns))

    for i in range(len(oi)):
        col = int(ot[i]//t_step)
        t = ot[i] - col*t_step

        if (t > t_lower):
            spikes_count[oi[i]][col] += 1

    return spikes_count