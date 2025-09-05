from tools_for_pauli_plus import *

basis_type = str(sys.argv[1]) ;
num_cycles = int(sys.argv[2]) ;

error_rate_excess = 0.03 ;


samples_map = [None, 100000, 160000, 280000, 500000, 800000, 1500000]
num_sim_samples = samples_map[num_cycles] ;

# set_error_rate_RO_to_Data = {"Z":0.035, "X": 0.035} ;
# error_rate_RO_to_Data = set_error_rate_RO_to_Data[basis_type] ;


set_T1 = { 0:44.7, 1:34.9, 2: 51.1, 3:45.3, 4:35.6, 5: 38.6, 6: 38.2,
          7: 34.4, 8: 44.4, 9: 55.0, 10: 50.8, 11: 48.6, 12: 39.5, 13: 38.6, 14: 37.2, 15: 37.9,
          16: 44.7, 17: 36.7, 18: 41.1, 19: 37.5, 20: 49.5, 21: 49.6, 22: 37.1, 23: 41.4, 24: 37.9,
          25: 45.4, 26: 42.3, 27: 57.8, 28: 35.3, 29: 34.2, 30: 33.5, 31: 39.9 }

set_Tphi = { 0: 21.0, 1: 55.1, 2: 13.8, 3: 17.0, 4: 29.6, 5: 26.1, 6: 63.4,
            7: 109.8, 8: 95.6, 9: 15.4, 10: 42.6, 11: 24.3, 12: 17.1, 13: 62.8, 14: 53.1, 15: 23.3,
            16: 17.9, 17: 18.0, 18: 41.4, 19: 14.7, 20: 64.1, 21: 69.2, 22: 44.5, 23: 52.5, 24: 54.2,
            25: 23.1, 26: 26.3, 27: 29.5, 28: 20.0, 29: 50.2, 30: 33.8, 31: 42.6 } ;

T1_ave = 41.8 ; Tphi_ave = 39.7


dict_channel_sq   = {}
dict_channel_tq   = {}
dict_channel_DD   = {}
dict_channel_idle = {} ;

for i in range(num_qubits):
    dict_channel_sq[i] = generate_channel_t(t_gate = 0.03, with_coherence="yes", with_heating="yes", T1 = set_T1[i], Tphi = set_Tphi[i] )
    dict_channel_tq[i] = generate_channel_t(t_gate = 0.105, with_coherence="yes", with_heating="yes",T1 =  set_T1[i], Tphi = set_Tphi[i] )
    dict_channel_DD[i] = generate_channel_t(t_gate = 0.92, with_coherence="yes", with_heating="yes", T1 = set_T1[i], Tphi = set_Tphi[i] )
    dict_channel_idle[i] = generate_channel_t(t_gate = 0.105, with_coherence="yes", with_heating="yes", T1 = set_T1[i], Tphi = set_Tphi[i] )


# num_level = 4 ;
# error_cz_xeb = 0.0098 ; # 0.0073 + 0.0025  
error_rate_idle = 0.0035
error_rate_sq = 0.0008 ;
error_rate_cz = 0.0073 ;
error_crosstalk = 0.0025 ;

# Physical error rates
data = loadmat("Exp_crosstalk_data/leak_matrix.mat"); leak_transit = data["leak_matrix"]
channel_cz = generate_channel_cz(leak_transit)  # CZ leakage, with higher-frequency qubit put in the front

ave_channel_sq = generate_channel_t( t_gate = 0.03, with_coherence="yes", with_heating="yes", T1 = T1_ave, Tphi = Tphi_ave )    
ave_channel_tq = generate_channel_t( t_gate = 0.105, with_coherence="yes", with_heating="yes", T1 = T1_ave, Tphi = Tphi_ave )     
ave_channel_DD = generate_channel_t( t_gate = 0.92, with_coherence="yes", with_heating="yes", T1 = T1_ave, Tphi = Tphi_ave )      
ave_channel_idle = generate_channel_t( t_gate = 0.105, with_coherence="yes", with_heating="yes", T1 = T1_ave, Tphi = Tphi_ave )   

# Depolarized error for compensation
error_rate_sq_com = error_rate_sq - (1 - ave_channel_sq.get_prob_from_to(0, 0, 0))
error_rate_cz_com = error_rate_cz - (2 - 2 * ave_channel_tq.get_prob_from_to(0, 0, 0) + 1 - channel_cz.get_prob_from_to(0, 0, 0) ) 
error_rate_idle_com = error_rate_idle - (1 - ave_channel_idle.get_prob_from_to(0, 0, 0))

#----------------------------------------------------------------------------------------------------------------------

# Circuit for measuring stabilizers
#----------------------------------------------------------------------------------------------------------------------
initial_pre = stim.Circuit()
initial_pre.append("R", Xcheck + data_L + data_R + Zcheck)
# initial_pre.append( "X_ERROR", Xcheck + data_L + data_R + Zcheck, error_rate_init )  
if basis_type == "X":
    initial_pre.append("H", data_L + data_R ) ;
#----------------------------------------------------------------------------------------------------------------------
SM_circuit = get_SM_circuit(error_rate_sq_com, error_rate_cz_com, error_rate_idle_com, error_crosstalk, remove_X_list, remove_Z_list, lin_order, \
                            data_qubits, Xchecks, Zchecks, nbs, sX, sZ)
#----------------------------------------------------------------------------------------------------------------------
circuit_DD = stim.Circuit()
circuit_DD.append("SQRT_X_DAG", data_L + data_R)
circuit_DD.append("SQRT_X", data_L + data_R)
circuit_DD.append("DEPOLARIZE1", data_L + data_R, error_rate_excess)
#----------------------------------------------------------------------------------------------------------------------
final = stim.Circuit()
if basis_type == "X":
    final.append("H", data_L + data_R ) ;
final.append("M", data_L + data_R )
#----------------------------------------------------------------------------------------------------------------------
Exp_circuit_logical = initial_pre + (SM_circuit + circuit_DD) * (num_cycles-1) + SM_circuit + final
#----------------------------------------------------------------------------------------------------------------------

# bind GPCs for the simulator
simulator = leaky.Simulator(num_qubits)

for targets in range(num_qubits):
    simulator.bind_leaky_channel(leaky.Instruction("Y", [targets]), dict_channel_sq[targets])
    simulator.bind_leaky_channel(leaky.Instruction("X", [targets]), dict_channel_sq[targets])
    simulator.bind_leaky_channel(leaky.Instruction("Z", [targets]), dict_channel_sq[targets])   
    simulator.bind_leaky_channel(leaky.Instruction("SQRT_Y", [targets]), dict_channel_sq[targets])

    #  qubit errors for DD
    simulator.bind_leaky_channel(leaky.Instruction("SQRT_X", [targets]), dict_channel_DD[targets])
    
    # qubit errors for CZ
    simulator.bind_leaky_channel(leaky.Instruction("S", [targets]), dict_channel_tq[targets])

     # qubit errors for iding
    simulator.bind_leaky_channel(leaky.Instruction("I", [targets]), dict_channel_idle[targets])
#----------------------------------------------------------------------------------------------------------------------
# CZ leakage
for target1, target2 in permutations(range(num_qubits), 2):
    simulator.bind_leaky_channel( leaky.Instruction("CZ", [target1, target2]), channel_cz )
#----------------------------------------------------------------------------------------------------------------------


# simulation
#----------------------------------------------------------------------------------------------------------------------
results_no_meas_error = simulator.sample_batch(Exp_circuit_logical, shots = num_sim_samples )
# index_no_leakage = np.all( (results_no_meas_error == 0) | (results_no_meas_error == 1), axis=1)
# np.sum(index_no_leakage)
# consider three-state readout errors
results = apply_transition_2d(num_databits, results_no_meas_error, check_three_meas_error, data_three_meas_error)
instance_no_leakage = np.all( (results == 0) | (results == 1), axis=1)
# rejecting instances with leakage detected
results_no_leakage = results[instance_no_leakage] ;
retained_num_samples = results_no_leakage.shape[0]

retain_data_fraction = retained_num_samples/num_sim_samples
# print(f"{retain_data_fraction*100}%")
#----------------------------------------------------------------------------------------------------------------------

if basis_type == "Z":
    
    syndrome_history_Z, final_logical_z_outcome = Z_transfer_meas_to_detection(hz, lz, num_cycles, results_no_leakage, num_databits)

    fname = './Numerical_data/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '.mat'
    savemat(fname, 
            { 'num_cycles': num_cycles, 
             'num_samples': num_sim_samples,
             'retain_data_fraction':retain_data_fraction,
             'retained_num_samples': retained_num_samples,
            'syndrome_history_Z': syndrome_history_Z, 'final_logical_z_outcome': final_logical_z_outcome })

if basis_type == "X":
    
    syndrome_history_X, final_logical_x_outcome = X_transfer_meas_to_detection(hx, lx, num_cycles, results_no_leakage, num_databits)

    fname = './Numerical_data/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '.mat'
    savemat(fname, 
            { 'num_cycles': num_cycles, 
             'num_samples': num_sim_samples,
             'retain_data_fraction':retain_data_fraction,
             'retained_num_samples': retained_num_samples,
            'syndrome_history_X': syndrome_history_X, 'final_logical_x_outcome': final_logical_x_outcome })





































































