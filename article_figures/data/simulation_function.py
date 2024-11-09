
import numpy as np 
import sys
from pyedlut import simulation_wrapper as pyedlut
import spikes_count_function as count

## Function that simulates the activity of MFs, GrCs and GoCs given 
## multiple parameters. To understand better the meaning of each parameter
## please run the main project notebooks from 01 to 05 where you will
## find a more detailed information about how simulations are run 
## and what are the relevant parameters involved

def simulation(dict, mf_activity):
    
    ## Receives a parameters dictionary and the MF input activity to start the simulations
    ## Returns the cells spikes times and indexes, as well as a matrix of GrCs firing rates
    ## and a matrix of spikes counts for all the cells (MFs, GrCs and GoCs) 
  
    f_mf = np.linspace(0.05, 0.95, 10)
    s = dict["seed"]
    poisson_seed = dict["poisson_seed"]
    n = dict["fraction"]
    fraction = f_mf[n]  #total active MF fraction of the simulation
    noise = dict["noise"]
    
    num_patterns = dict["num_patterns"] ## number of patterns
    firing_rate = dict["firing_rate"]
    goc_grc_w = 0.5
    mf_grc_w_0 = 0.00 # Total weight that reaches each GRC 
    mf_grc_w = 4.00
    n_mf_grc = 4 # Number of MF connected to each GRC
    mf_goc_w = dict["mf_goc_w"]
    grc_goc_w = dict["grc_goc_w"]


    sys.path.insert(1, '../../data/structure')
    with open('../../data/structure/seed' + str(s) + '/glos.npy', 'rb') as f:
        glos = np.load(f)
    with open('../../data/structure/seed' + str(s) + '/grcs.npy', 'rb') as f:
        grcs = np.load(f)
    with open('../../data/structure/seed' + str(s) + '/gocs.npy', 'rb') as f:
        gocs = np.load(f)    
    with open('../../data/structure/seed' + str(s) + '/conn_mat_glos_to_grc.npy', 'rb') as f:
        conn_mat_glos_to_grc = np.load(f)    
    with open('../../data/structure/seed' + str(s) + '/conn_mat_glos_to_goc.npy', 'rb') as f:
        conn_mat_glos_to_goc = np.load(f)
    with open('../../data/structure/seed' + str(s) + '/conn_mat_grcs_to_goc.npy', 'rb') as f:
        conn_mat_grcs_to_goc = np.load(f)
    with open('../../data/structure/seed' + str(s) + '/conn_mat_goc_to_grcs.npy', 'rb') as f:
        conn_mat_goc_to_grcs = np.load(f) 

    ## Number of cells for each population
    n_inputs = glos.shape[0]
    n_hidden_neurons = grcs.shape[0]
    n_outputs = gocs.shape[0]

    # Time steps in seconds 
    edlut_time_step = 5e-4
    duration_pattern = dict["duration_pattern"]
    interval = dict["interval"] ## spikes count interval

    # Simulation time in seconds
    stop_simulation_at = num_patterns * duration_pattern 

    # LIF neuron parameters
    default_neuron_params = {
    'c_m': 250.0,
    'e_exc': 0.0,
    'e_inh': -85.0,
    'e_leak': -65.0,
    'g_leak': 25.0,
    'tau_exc': 5.0,
    'tau_inh': 10.0,
    'tau_nmda': 20.0,
    'tau_ref': 1.0,
    'v_thr': -40.0,
    'int_meth': None,
    }

    poisson_generator_params = {
    'frequency': firing_rate,
    }

    hidden_neuron_params = { 
    **default_neuron_params,
    'c_m': 2.0,
    'e_exc': 0.0,
    'e_inh': -65.0,
    'e_leak': -65.0,
    'g_leak': 0.2,
    'tau_exc': 0.5,
    'tau_inh': 10.0,
    'tau_nmda': 40.0,
    'tau_ref': 1.5,
    'v_thr': -40.0,   
    }

    # Parámetros de las GOC
    output_neuron_params = {
    **default_neuron_params,        
    'c_m': 50.0,
    'g_leak': 10.0,
    'v_thr': -50.0,
    } 


    default_synapse_params = {
    'weight': 0.006,    #Initial weight
    'max_weight': 0.020, #Max weight
    'type': 0,
    'delay': 0.001,
    'wchange': -1,
    'trigger_wchange': -1,
    }


    ## input patterns (binary vectors)
    input_patterns = np.zeros((n_inputs, num_patterns))
    input_patterns = np.tile(mf_activity, reps = 1)


    # Declare the simulation object
    simulation = pyedlut.PySimulation_API()
    simulation.SetRandomGeneratorSeed(poisson_seed)

    # Creación del método de integración y de la regla de aprendizaje
    integration_method = pyedlut.PyModelDescription(model_name='RK4', params_dict={'step': edlut_time_step})
    default_neuron_params['int_meth'] = integration_method
    hidden_neuron_params['int_meth'] = integration_method

    # Parámetros según el tipo de sinapsis
    mf_grc_synapse_params = {**default_synapse_params,'weight': mf_grc_w_0/n_mf_grc,'max_weight': mf_grc_w_0/n_mf_grc, 'type': 0}

    #################### Creating neuron layers ########################################
    # MFs layer
    input_poisson_generator_layer = simulation.AddNeuronLayer(
        num_neurons = n_inputs,
        model_name = 'PoissonGeneratorDeviceVector',
        param_dict = poisson_generator_params,
        log_activity = False,
        output_activity = True
    )

    # GRCs layer
    hidden_layer = simulation.AddNeuronLayer(
        num_neurons = n_hidden_neurons,
        model_name = 'LIFTimeDrivenModel',
        param_dict = hidden_neuron_params,
        log_activity = False,
        output_activity = False
    )

    #################### Connections from MF to GRC ################################

    sources2 = []
    hidden2 = []

    for i in range(conn_mat_glos_to_grc.shape[1]): 
        srcs = np.where(conn_mat_glos_to_grc[:,i] == 1)[0].tolist()
        sources2.append(srcs)
        hid = i + n_inputs
        hidden2.append([hid]*len(srcs))


    sources = []
    hidden = []

    ## saving the indexes of all the MFs connected to each GRC
    for sublist in sources2:
        for item in sublist:
            sources.append(item)

    for sublist in hidden2:
        for item in sublist:
            hidden.append(item)

    # MF -> GRC excitatory synapsis 
    _ = simulation.AddSynapticLayer(
        source_list = sources,
        target_list = hidden,
        param_dict = mf_grc_synapse_params,
    )    

    simulation.Initialize()

    ################ SET FINAL PARAMETER IN PoissonGeneratorDeviceVector #####################

    for i in range(n_inputs):
        poisson_params = {'frequency': firing_rate * input_patterns[i, 0]} # noise for MFs in the inner cube 
        simulation.SetSpecificNeuronParams(input_poisson_generator_layer[i], poisson_params)

    ####################-##### Run the simulation step-by-step ##################################

    # Run the simulation step-by-step

    total_simulation_time = stop_simulation_at
    simulation_bin = duration_pattern

    # we run the simulation step by step (pattern by pattern)
    j = 1
    for sim_time in np.arange(0.0 + simulation_bin, total_simulation_time + simulation_bin - 0.000001, simulation_bin):

        simulation.RunSimulation(sim_time)

        if j < (num_patterns):

            #Update all poissons every timebin
            for i in range(n_inputs):
                poisson_params = {'frequency': firing_rate * input_patterns[i,j]}
                simulation.SetSpecificNeuronParams(input_poisson_generator_layer[i], poisson_params)

        j += 1

    # Retrieve output spike activity
    output_times, output_index = simulation.GetSpikeActivity()
    ot0 = np.array(output_times)
    oi0 = np.array(output_index)

    seed_noise = dict["seed_noise"]
    stop_simulation_at = num_patterns * duration_pattern 
    ## input patterns (binary vectors)
    input_patterns = np.zeros((n_inputs,num_patterns))

    # Declare the simulation object
    simulation = pyedlut.PySimulation_API()
    simulation.SetRandomGeneratorSeed(poisson_seed + seed_noise + 1)

    # Creación del método de integración y de la regla de aprendizaje
    integration_method = pyedlut.PyModelDescription(model_name='RK4', params_dict={'step': edlut_time_step})
    default_neuron_params['int_meth'] = integration_method
    hidden_neuron_params['int_meth'] = integration_method

    # Parámetros según el tipo de sinapsis

    mf_grc_synapse_params = {**default_synapse_params,'weight':mf_grc_w_0/n_mf_grc,'max_weight': mf_grc_w_0/n_mf_grc, 'type': 0}

    #################### Creating neuron layers ########################################

    # MFs layer
    input_poisson_generator_layer = simulation.AddNeuronLayer(
        num_neurons = n_inputs,
        model_name = 'PoissonGeneratorDeviceVector',
        param_dict = poisson_generator_params,
        log_activity = False,
        output_activity = True
    )    

    # GRCs layer
    hidden_layer = simulation.AddNeuronLayer(
        num_neurons = n_hidden_neurons,
        model_name = 'LIFTimeDrivenModel',
        param_dict = hidden_neuron_params,
        log_activity = False,
        output_activity = False
    )


    #################### Connections from MF to GRC ################################

    sources2 = []
    hidden2 = []

    for i in range(conn_mat_glos_to_grc.shape[1]): 
        srcs = np.where(conn_mat_glos_to_grc[:,i] == 1)[0].tolist()
        sources2.append(srcs)
        hid = i + n_inputs
        hidden2.append([hid]*len(srcs))


    sources = []
    hidden = []

    ## saving the indexes of all the MFs connected to each GRC
    for sublist in sources2:
        for item in sublist:
            sources.append(item)

    for sublist in hidden2:
        for item in sublist:
            hidden.append(item)

    # MF -> GRC excitatory synapsis 
    weights = simulation.AddSynapticLayer(
    source_list = sources,
    target_list = hidden,
    param_dict = mf_grc_synapse_params,
    )    

    simulation.Initialize()


    # Activating all MFs in order to generate spikes in all of them
    for i in range(n_inputs):
            input_patterns[i,:] = 1.0

    ################ SET FINAL PARAMETER IN PoissonGeneratorDeviceVector #####################

    for i in range(n_inputs):

        poisson_params = {'frequency': firing_rate * input_patterns[i, 0]} # noise for MFs in the inner cube 
        simulation.SetSpecificNeuronParams(input_poisson_generator_layer[i], poisson_params)

        ####################-##### Run the simulation step-by-step ##################################

    # Run the simulation step-by-step

    total_simulation_time = stop_simulation_at
    simulation_bin = duration_pattern



    # we run the simulation step by step (pattern by pattern)
    j = 1

    for sim_time in np.arange(0.0 + simulation_bin, total_simulation_time + simulation_bin - 0.000001, simulation_bin):

        simulation.RunSimulation(sim_time)

        if j < (num_patterns):

            #Update all poissons every timebin
            for i in range(n_inputs):
                poisson_params = {'frequency':firing_rate * input_patterns[i,j]}
                simulation.SetSpecificNeuronParams(input_poisson_generator_layer[i], poisson_params)

        j += 1


    # Retrieve output spike activity
    output_times, output_index = simulation.GetSpikeActivity()
    ot_k = np.array(output_times)
    oi_k = np.array(output_index)



    #########################################################################################
    ### MERGE OF S0 AND S_K #################################################################
    #########################################################################################



    # Declare the simulation object
    simulation = pyedlut.PySimulation_API()

    # Creación del método de integración y de la regla de aprendizaje
    integration_method = pyedlut.PyModelDescription(model_name='RK4', params_dict={'step': edlut_time_step})
    default_neuron_params['int_meth'] = integration_method
    hidden_neuron_params['int_meth'] = integration_method
    output_neuron_params['int_meth'] = integration_method

    # Parámetros según el tipo de sinapsis

    mf_grc_synapse_params = {**default_synapse_params,'weight': mf_grc_w/n_mf_grc,'max_weight': mf_grc_w/n_mf_grc, 'type': 0}
    mf_goc_synapse_params = {**default_synapse_params,'weight': mf_goc_w,'max_weight': mf_goc_w, 'type': 0}
    grc_goc_synapse_params = {**default_synapse_params,'weight': grc_goc_w,'max_weight': grc_goc_w, 'type': 0}
    goc_grc_synapse_params = {**default_synapse_params,'weight': goc_grc_w,'max_weight': goc_grc_w, 'type': 1}

    #################### Creating neuron layers ########################################

    # MFs layer
    input_layer = simulation.AddNeuronLayer(
        num_neurons = n_inputs,
        model_name = 'InputSpikeNeuronModel',
        param_dict = {},
        log_activity = False,
        output_activity = True
    )

    # GRCs layer
    hidden_layer = simulation.AddNeuronLayer(
        num_neurons = n_hidden_neurons,
        model_name = 'LIFTimeDrivenModel',
        param_dict = hidden_neuron_params,
        log_activity = False,
        output_activity = True
    )

    # GoCs layer
    output_layer = simulation.AddNeuronLayer(
        num_neurons = n_outputs,
        model_name = 'LIFTimeDrivenModel',
        param_dict = output_neuron_params,
        log_activity = False,
        output_activity = True
    )

    #################### Connections from MF to GRC ################################

    sources2 = []
    hidden2 = []

    for i in range(conn_mat_glos_to_grc.shape[1]): 
        srcs = np.where(conn_mat_glos_to_grc[:,i] == 1)[0].tolist()
        sources2.append(srcs)
        hid = i + n_inputs
        hidden2.append([hid]*len(srcs))


    sources = []
    hidden = []

    ## saving the indexes of all the MFs connected to each GRC
    for sublist in sources2:
        for item in sublist:
            sources.append(item)

    for sublist in hidden2:
        for item in sublist:
            hidden.append(item)

    # MF -> GRC excitatory synapsis 
    weights = simulation.AddSynapticLayer(
    source_list = sources,
    target_list = hidden,
    param_dict = mf_grc_synapse_params,
    )    

    ######################## Connections from MF to GOC ###############################

    sources2 = []
    targets2 = []

    for i in range(conn_mat_glos_to_goc.shape[1]): 
        srcs = np.where(conn_mat_glos_to_goc[:,i] == 1)[0].tolist()
        sources2.append(srcs)
        tar = i + n_inputs + n_hidden_neurons
        targets2.append([tar]*len(srcs))

    sources = []
    targets = []

    for sublist in sources2:
        for item in sublist:
            sources.append(item)    

    for sublist in targets2:
        for item in sublist:
            targets.append(item)


    # MF -> GOC excitatory synapsis 
    _ = simulation.AddSynapticLayer(
    source_list = sources,
    target_list = targets,
    param_dict = mf_goc_synapse_params,
    )

    #################### Connections from to GRC to GOC #################################

    hidden2 = []
    targets2 = []

    for i in range(conn_mat_grcs_to_goc.shape[1]): 
        hids = np.where(conn_mat_grcs_to_goc[:,i] == 1)[0].tolist()
        hidden2.append([hid + n_inputs for hid in hids])
        tar = i + n_inputs + n_hidden_neurons
        targets2.append([tar]*len(hids))

    hiddens = []
    targets = []

    for sublist in hidden2:
        for item in sublist:
            hiddens.append(item)    

    for sublist in targets2:
        for item in sublist:
            targets.append(item)    

    #  GRC -> GOC excitatory synapsis 
    _ = simulation.AddSynapticLayer(
    source_list = hiddens,
    target_list = targets,
    param_dict = grc_goc_synapse_params,
    )

    #################### Connections from to GOC to GRC ##################################

    hidden2 = []
    targets2 = []

    for i in range(conn_mat_goc_to_grcs.shape[1]): 
        hids = np.where(conn_mat_goc_to_grcs[:,i] == 1)[0].tolist()
        hidden2.append([hid + n_inputs for hid in hids])
        tar = i + n_inputs + n_hidden_neurons
        targets2.append([tar]*len(hids))

    hiddens = []
    targets = []

    for sublist in hidden2:
        for item in sublist:
            hiddens.append(item)    

    for sublist in targets2:
        for item in sublist:
            targets.append(item)    

    # GOC -> GRC inhibitory synapsis 
    _ = simulation.AddSynapticLayer(
    source_list = targets,
    target_list = hiddens,
    param_dict = goc_grc_synapse_params,
    )

    simulation.Initialize()


    dt = duration_pattern
    np.random.seed(seed_noise)    
    time = 0.0 ## silent MF percentage used to introduce noise 
    n_noise = round(noise*n_inputs) ## we calculate the number 
    noise_indexes = np.random.choice(range(n_inputs), n_noise, replace = False) ## indexes of the mfs that we will flip their activity
    n_active = round(fraction*len(noise_indexes))
    active_indexes = np.random.choice(noise_indexes, n_active, replace = False)

    tim = (ot0>=time) * (ot0<=(time+dt)) 
    ot_new0 = ot0[tim]
    oi_new0 = oi0[tim]
    for elemento in noise_indexes: 
        positions = list(np.where(oi_new0 == elemento)[0])
        if positions:
            oi_new0 = np.delete(oi_new0, positions)
            ot_new0 = np.delete(ot_new0, positions)

    tim = (ot_k>=time) * (ot_k<=(time+dt)) 
    ot_newk = ot_k[tim]
    oi_newk = oi_k[tim]
    for elemento in active_indexes: 
        positions = list(np.where(oi_k[tim] == elemento)[0])
        if positions:
            oi_new0 = np.concatenate((oi_new0, oi_newk[positions]))
            ot_new0 = np.concatenate((ot_new0, ot_newk[positions]))

    active_indexes_total = []
    active_indexes_total.append(active_indexes)

    # Add external spike activity (times and neuron indexes)
    simulation.AddExternalSpikeActivity(ot_new0, oi_new0)

    # Run the simulation step-by-step

    total_simulation_time = stop_simulation_at
    simulation_bin = duration_pattern


    # we run the simulation step by step (pattern by pattern)
    j = 1
    for sim_time in np.arange(0.0 + simulation_bin, total_simulation_time + simulation_bin - 0.000001, simulation_bin):

        simulation.RunSimulation(sim_time)

        if j < (num_patterns):

            seed_noise += 1    
            np.random.seed(seed_noise)

            ## silent MF percentage used to introduce noise 
            n_noise = round(noise*n_inputs) ## we calculate the number 
            noise_indexes = np.random.choice(range(n_inputs), n_noise, replace = False) ## indexes of the mfs that we will flip their activity
            n_active = round(fraction*len(noise_indexes))
            active_indexes = np.random.choice(noise_indexes, n_active, replace = False)


            tim = (ot0>=sim_time) * (ot0<=(sim_time+dt)) 
            ot_new0 = ot0[tim]
            oi_new0 = oi0[tim]

            for elemento in noise_indexes: 
                positions = list(np.where(oi_new0 == elemento)[0])
                if positions:
                    oi_new0 = np.delete(oi_new0, positions)
                    ot_new0 = np.delete(ot_new0, positions)

            tim = (ot_k>=sim_time) * (ot_k<=(sim_time+dt)) 
            ot_newk = ot_k[tim]
            oi_newk = oi_k[tim]

            for elemento in active_indexes: 

                positions = list(np.where(oi_k[tim] == elemento)[0])

                if positions:
                    oi_new0 = np.concatenate((oi_new0, oi_newk[positions]))
                    ot_new0 = np.concatenate((ot_new0, ot_newk[positions]))

            # Add external spike activity (times and neuron indexes)
            simulation.AddExternalSpikeActivity(ot_new0, oi_new0)

        j += 1


    seed_noise += 1


    # Retrieve output spike activity
    output_times, output_index = simulation.GetSpikeActivity()
    ot = np.array(output_times)
    oi = np.array(output_index)

    step = dict["step"]
    t_step = duration_pattern / step
    t_lower = duration_pattern - interval
    matrix = count.spikes_count(n_inputs, n_hidden_neurons, n_outputs, oi, ot, t_step, t_lower, total_simulation_time)

    matrix_fr = matrix[n_inputs:(n_inputs + n_hidden_neurons),:]/t_step

    return oi, ot, matrix_fr, matrix

 