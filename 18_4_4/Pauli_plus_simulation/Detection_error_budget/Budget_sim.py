# 
import os
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../.."))
from tools_for_pauli_plus import *


basis_type = str(sys.argv[1]) ;
error_rate_RO_to_Data = float(sys.argv[2]) ;
budget_component = str(sys.argv[3])  ;


num_cycles = 6 ;
num_sim_samples = 150000 ;


# num_level = 4 ;
# error_cz_xeb = 0.0098;  # 0.0073 + 0.0025  
error_rate_init = 0.003 ;
error_rate_idle = 0.0035
error_rate_sq = 0.0008 ;
error_rate_cz = 0.0073 ;
error_crosstalk = 0.0025 ;

# Physical error rates
data = loadmat("../Exp_crosstalk_data/leak_matrix.mat"); leak_transit = data["leak_martix"]
channel_cz = generate_channel_cz(leak_transit)  # CZ leakage, with higher-frequency qubit put in the front

channel_sq = generate_channel_t(0.03)       # 30ns single-qubit gate
channel_tq = generate_channel_t(0.105)      # 105ns two-qubit gate
channel_DD = generate_channel_t(0.92)       # 920ns DD
channel_idle = generate_channel_t(0.105)    # 105ns idle

# Depolarized error for compensation
error_rate_sq_com = error_rate_sq - (1 - channel_sq.get_prob_from_to(0, 0, 0))
error_rate_cz_com = error_rate_cz - (2 - 2 * channel_tq.get_prob_from_to(0, 0, 0) + 1 - channel_cz.get_prob_from_to(0, 0, 0) ) 
error_rate_idle_com = error_rate_idle - (1 - channel_idle.get_prob_from_to(0, 0, 0))
#----------------------------------------------------------------------------------------------------------------------


if budget_component == "CZ" :
    error_rate_cz_com = 0
    channel_tq = generate_channel_t(0)
    channel_cz = generate_channel_t(0)

if budget_component == "crosstalk" :
    error_crosstalk = 0 

if budget_component == "SQ" :
    error_rate_sq_com = 0
    channel_sq = generate_channel_t(0)

if budget_component == "DD" :
    channel_DD = generate_channel_t(0)

if budget_component == "idle" :
    error_rate_idle_com = 0
    channel_idle = generate_channel_t(0)

if budget_component == "RO_Data" :
    error_rate_RO_to_Data = 0 

if budget_component == "initial_readout" :
    error_rate_init = 0 ;
    num_sim_samples = 30000 ;
    check_three_meas_error = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1  ],
                                       [0, 0, 1]]) ;
    
    data_three_meas_error = np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1],
                                      [0, 0, 1]]) ;
#----------------------------------------------------------------------------------------------------------------------

# Circuit for measuring stabilizers
initial_pre = stim.Circuit()
initial_pre.append("R", Xcheck + data_L + data_R + Zcheck)
initial_pre.append( "X_ERROR", Xcheck + data_L + data_R + Zcheck, error_rate_init )  
if basis_type == "X":
    initial_pre.append("H", data_L + data_R ) ;
#----------------------------------------------------------------------------------------------------------------------
SM_circuit = get_SM_circuit(error_rate_sq_com, error_rate_cz_com, error_rate_idle_com, error_crosstalk, remove_X_list, remove_Z_list, lin_order, \
                            data_qubits, Xchecks, Zchecks, nbs, sX, sZ)
#----------------------------------------------------------------------------------------------------------------------
circuit_DD = stim.Circuit()
circuit_DD.append("SQRT_X_DAG", data_L + data_R)
circuit_DD.append("SQRT_X", data_L + data_R)
circuit_DD.append("DEPOLARIZE1", data_L + data_R, error_rate_RO_to_Data)
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
    simulator.bind_leaky_channel(leaky.Instruction("Y", [targets]), channel_sq)
    simulator.bind_leaky_channel(leaky.Instruction("X", [targets]), channel_sq)
    simulator.bind_leaky_channel(leaky.Instruction("Z", [targets]), channel_sq)   
    simulator.bind_leaky_channel(leaky.Instruction("SQRT_Y", [targets]), channel_sq)

    #  qubit errors for DD
    simulator.bind_leaky_channel(leaky.Instruction("SQRT_X", [targets]), channel_DD)
    
    # qubit errors for CZ
    simulator.bind_leaky_channel(leaky.Instruction("S", [targets]), channel_tq)

     # qubit errors for iding
    simulator.bind_leaky_channel(leaky.Instruction("I", [targets]), channel_idle)
#----------------------------------------------------------------------------------------------------------------------
# CZ leakage
for target1, target2 in permutations(range(num_qubits), 2):
    simulator.bind_leaky_channel( leaky.Instruction("CZ", [target1, target2]), channel_cz )
#----------------------------------------------------------------------------------------------------------------------


# simulation
#----------------------------------------------------------------------------------------------------------------------
results_no_meas_error = simulator.sample_batch(Exp_circuit_logical, shots = num_sim_samples )
index_no_leakage = np.all( (results_no_meas_error == 0) | (results_no_meas_error == 1), axis=1)
np.sum(index_no_leakage)
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

    fname = './Numerical_data/' + 'Logical_Z_' + 'budget_' + budget_component + '.mat'
    savemat(fname, 
            { 'num_samples': num_sim_samples,
             'retain_data_fraction':retain_data_fraction,
             'retained_num_samples': retained_num_samples,
            'syndrome_history_Z': syndrome_history_Z, 'final_logical_z_outcome': final_logical_z_outcome })

if basis_type == "X":
    
    syndrome_history_X, final_logical_x_outcome = X_transfer_meas_to_detection(hx, lx, num_cycles, results_no_leakage, num_databits)

    fname = './Numerical_data/' + 'Logical_X_' + 'budget_' + budget_component + '.mat'
    savemat(fname, 
            { 'num_samples': num_sim_samples,
             'retain_data_fraction':retain_data_fraction,
             'retained_num_samples': retained_num_samples,
            'syndrome_history_X': syndrome_history_X, 'final_logical_x_outcome': final_logical_x_outcome })













