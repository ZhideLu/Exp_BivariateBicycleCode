from tabulate import tabulate
import numpy as np
from mip import Model, xsum, minimize, BINARY
from mip import OptimizationStatus
from bposd.css import css_code
from ldpc import bposd_decoder
import itertools
from scipy.sparse import coo_matrix, hstack 
from ldpc import mod2
import pickle
import sys
import os
from scipy.io import savemat
from scipy.io import loadmat

sys.path.append(os.path.abspath("..")) 
from functions_BB_code import error_decoding

# number of syndrome measurement cycles (not including final data qubit measurements)
num_cycles = int(sys.argv[1])


num_L = 9 ; num_R = 9 ;
k = 4 ; n2 = 9 ;
index_Z_check = [0,1,2,5,6,7,8] ;
index_X_check = [0,1,3,4,5,6,7] ;

with open('Set_decoder_para_18_4_4.pkl', 'rb') as f:
    Set_decoder_para_18_4_4 = pickle.load(f)

k = 4 ; n2 = 9 ;

channel_probsX = Set_decoder_para_18_4_4[f"channel_probsX_{num_cycles}"]
HX = Set_decoder_para_18_4_4[f"HX_{num_cycles}"]  ;
HdecX = Set_decoder_para_18_4_4[f"HdecX_{num_cycles}"] ;

channel_probsZ = Set_decoder_para_18_4_4[f"channel_probsZ_{num_cycles}"]
HZ = Set_decoder_para_18_4_4[f"HZ_{num_cycles}"]  ;
HdecZ = Set_decoder_para_18_4_4[f"HdecZ_{num_cycles}"] ;

# Setup BP-OSD decoder parameters
my_bp_method = "ms"
my_max_iter = 10000
my_osd_method = "osd_cs"
my_osd_order = 7
my_ms_scaling_factor = 0

bpdX=bposd_decoder(
    HdecX, #the parity check matrix
    channel_probs=channel_probsX, #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=my_max_iter, #the maximum number of iterations for BP)
    bp_method=my_bp_method,
    ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method=my_osd_method, #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=my_osd_order #the osd search depth
    )

bpdZ=bposd_decoder(
    HdecZ, #the parity check matrix
    channel_probs=channel_probsZ, #assign error_rate to each qubit. This will override "error_rate" input variable
    max_iter=my_max_iter, #the maximum number of iterations for BP)
    bp_method=my_bp_method,
    ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
    osd_order=my_osd_order #the osd search depth
    )


# Logical Z data
# data processing
# remove_leakage_
with open (f'Experimental_data/remove_leakage_ldpc_test_result_logical_Z_cycle_num={num_cycles}','rb') as f:
    data_z = pickle.load(f)
    
# with open (f'Experimental_data/with_leakage/ldpc_test_result_logical_Z_cycle_num={num_cycles}','rb') as f:
#     data_z = pickle.load(f)

# Z check matrix and logical operators for the BB code [[18,4,4]] 
hz = Set_decoder_para_18_4_4["hz"] ;
lz = Set_decoder_para_18_4_4["lz"] ;

# data qubit的初态 共有 M 个初态
random_initial_state_z = data_z['random_initial_state']

num_initial_z = random_initial_state_z.shape[0]
# print(random_initial_state_z[0])

# 按照L0...L8,R0...R8的顺序排列
for iter in range( num_initial_z ) :
    random_initial_state_z[iter] = dict(sorted(random_initial_state_z[iter].items(), key=lambda x: (x[0][2:])))

#将 dict 转化为 list
list_initial_z = [] ;
for iter in range( num_initial_z  ) :
    list_initial_z.append( list(random_initial_state_z[iter].values()) )

##---------------------------------------------------------------------------------------------------------------------     
# record the Z parity check outcomes and the logical Z outcomes
dict_cycle_z_parity = {} ;
dict_cycle_z_devents = {}
for i in range(1, num_cycles+1):
    dict_cycle_z_parity[f'cycle_{i}'] = [] ;
    dict_cycle_z_devents[f'cycle_{i}'] = [] ;

dict_cycle_z_parity['final'] = [] ;
dict_cycle_z_devents['final'] = [] ;

final_logical_z_outcome = [] ;

for s in range(num_initial_z):

    #第 s 个初始态
    initial_z_parity = hz @ np.array(list_initial_z[s]).T % 2

    num_trials_z = len(data_z['states']['cycle_1'][f'q_Z{1}'][s]) ; 
    
    initial_logical_z_outcome = (lz @ list_initial_z[s] ) % 2

    for p in range(num_trials_z):
    
        dict_cycle_z_parity['cycle_1'].append( np.array( [data_z['states']['cycle_1'][f'q_Z{i}'][s][p] for i in index_Z_check  ] )  )
    
        for cycle in range(2, 1+num_cycles):
            current = np.array([data_z['states'][f'cycle_{cycle}'][f'q_Z{i}'][s][p] for i in index_Z_check  ]) ;
            former = np.array([data_z['states'][f'cycle_{cycle-1}'][f'q_Z{i}'][s][p] for i in index_Z_check  ]) ;    
            dict_cycle_z_parity[f'cycle_{cycle}'].append( (  current + former ) %2  )
        
        final_state_z = [ data_z['final_cycle_states'][f'q_L{i}'][s][p]  for i in range(num_L) ] + \
                     [ data_z['final_cycle_states'][f'q_R{i}'][s][p]  for i in range(num_R) ]  ;
    
        dict_cycle_z_parity['final'].append( hz @ np.array(final_state_z).T % 2 ) ;   
    
        final_logical_z_outcome.append(   (lz @ final_state_z - initial_logical_z_outcome) % 2   )

        # record the deventions, a error devent occur if the current parity is different from the former one
##---------------------------------------------------------------------------------------------------------------------         
        dict_cycle_z_devents['cycle_1'].append( (dict_cycle_z_parity['cycle_1'][-1]  + initial_z_parity) %2  )
        
        for cycle in range(2, num_cycles+1):

            dict_cycle_z_devents[f'cycle_{cycle}'].append( (dict_cycle_z_parity[f'cycle_{cycle}'][-1] + \
                                                            dict_cycle_z_parity[f'cycle_{cycle-1}'][-1]) %2 ) ;

        dict_cycle_z_devents['final'].append( (dict_cycle_z_parity['final'][-1] + dict_cycle_z_parity[f'cycle_{num_cycles}'][-1]) %2  )
##---------------------------------------------------------------------------------------------------------------------     


# error decoding
syndrome_history_Z = [np.hstack((arr)) for arr in zip(*dict_cycle_z_devents.values())] ;
# final_logical_z_outcome
num_detect_z = len(syndrome_history_Z[0]) ;
num_instance_z = len(syndrome_history_Z) ;


# correct errors for logical Z (Z checks)
Z_no_correction_good_trials, Z_no_correction_good_list, Z_good_trials, Z_good_list =     \
            error_decoding(num_instance_z, k, bpdZ, HZ, syndrome_history_Z, final_logical_z_outcome )


print("no error correction for logical Z basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-Z_no_correction_good_trials/num_instance_z)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-Z_no_correction_good_list/num_instance_z)
print("\n")
print("error correction for logical Z basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-Z_good_trials/num_instance_z)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-Z_good_list/num_instance_z)


fname1 = './Experimental_results/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '.mat'

# fname1 = './Experimental_results/with_leakage/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '.mat'

savemat(fname1, 
        { 'num_cycles': num_cycles, 
         'num_instance_z':num_instance_z,
        'dict_cycle_z_devents': dict_cycle_z_devents, 
        'logical_error_probability':  1-Z_good_trials/num_instance_z,
        'logical_error_probability_per_qubit': 1-Z_good_list/num_instance_z
        })


# Logical X data
# data processing
# remove_leakage_
with open (f'Experimental_data/remove_leakage_ldpc_test_result_logical_X_cycle_num={num_cycles}','rb') as f:
    data_x = pickle.load(f)

# with open (f'Experimental_data/with_leakage/ldpc_test_result_logical_X_cycle_num={num_cycles}','rb') as f:
#     data_x = pickle.load(f)

# X check matrix for the BB code [[18,4,4]] 
hx = Set_decoder_para_18_4_4["hx"] ;
lx = Set_decoder_para_18_4_4["lx"] ;

# data qubit的初态
random_initial_state_x = data_x['random_initial_state']
num_initial_x = random_initial_state_x.shape[0]
# print(random_initial_state_x[0])

# 按照L0...L8,R0...R8的顺序排列
for iter in range( num_initial_x ) :
    random_initial_state_x[iter] = dict(sorted(random_initial_state_x[iter].items(), key=lambda x: (x[0][2:])))

#将 dict 转化为 list
list_initial_x = [] ;
for iter in range( num_initial_x  ) :
    list_initial_x.append( list(random_initial_state_x[iter].values()) )

##---------------------------------------------------------------------------------------------------------------------  
# record the X parity check outcomes and the logical Z outcomes
dict_cycle_x_parity = {} ;
dict_cycle_x_devents = {}
for i in range(1, num_cycles+1):
    dict_cycle_x_parity[f'cycle_{i}'] = [] ;
    dict_cycle_x_devents[f'cycle_{i}'] = [] ;

dict_cycle_x_parity['final'] = [] ;
dict_cycle_x_devents['final'] = [] ;

final_logical_x_outcome = [] ;

for s in range(num_initial_x):

    #第 s 个初始态
    initial_x_parity = hx @ np.array(list_initial_x[s]).T % 2

    num_trials_x = len(data_x['states']['cycle_1'][f'q_X{1}'][s]) ; 
    
    initial_logical_x_outcome = (lx @ list_initial_x[s] ) % 2

    for p in range(num_trials_x):
    
        dict_cycle_x_parity['cycle_1'].append( np.array( [data_x['states']['cycle_1'][f'q_X{i}'][s][p] for i in index_X_check  ] )  )
    
        for cycle in range(2, 1+num_cycles):
            current = np.array([data_x['states'][f'cycle_{cycle}'][f'q_X{i}'][s][p] for i in index_X_check  ]) ;
            former = np.array([data_x['states'][f'cycle_{cycle-1}'][f'q_X{i}'][s][p] for i in index_X_check  ]) ;    
            dict_cycle_x_parity[f'cycle_{cycle}'].append( (  current + former ) %2  )
        
        final_state_x = [ data_x['final_cycle_states'][f'q_L{i}'][s][p]  for i in range(num_L) ] + \
                     [ data_x['final_cycle_states'][f'q_R{i}'][s][p]  for i in range(num_R) ]  ;
    
        dict_cycle_x_parity['final'].append( hx @ np.array(final_state_x).T % 2 ) ;   
    
        final_logical_x_outcome.append(   (lx @ final_state_x - initial_logical_x_outcome) % 2   )

        # record the deventions, a error devent occur if the current parity is different from the former one
##------------------------------------------------------------------------------------------------------------------         
        dict_cycle_x_devents['cycle_1'].append( (dict_cycle_x_parity['cycle_1'][-1]  + initial_x_parity) %2  )
        
        for cycle in range(2, num_cycles+1):

            dict_cycle_x_devents[f'cycle_{cycle}'].append( (dict_cycle_x_parity[f'cycle_{cycle}'][-1] + \
                                                            dict_cycle_x_parity[f'cycle_{cycle-1}'][-1]) %2 ) ;

        dict_cycle_x_devents['final'].append( (dict_cycle_x_parity['final'][-1] + dict_cycle_x_parity[f'cycle_{num_cycles}'][-1]) %2  )
##---------------------------------------------------------------------------------------------------------------------  

# error decoding
syndrome_history_X = [np.hstack((arr)) for arr in zip(*dict_cycle_x_devents.values())] ;
# final_logical_x_outcome
num_detect_x = len(syndrome_history_X[0]) ;
num_instance_x = len(syndrome_history_X) ;

# correct errors for logical X (X checks)
X_no_correction_good_trials, X_no_correction_good_list, X_good_trials, X_good_list =     \
            error_decoding(num_instance_x, k, bpdX, HX, syndrome_history_X, final_logical_x_outcome )


print("no error correction for logical X basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-X_no_correction_good_trials/num_instance_x)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-X_no_correction_good_list/num_instance_x)
print("\n")
print("error correction for logical X basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-X_good_trials/num_instance_x)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-X_good_list/num_instance_x)


fname2 = './Experimental_results/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '.mat'

# fname2 = './Experimental_results/with_leakage/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '.mat'

savemat(fname2, 
        { 'num_cycles': num_cycles, 
          'num_instance_x':num_instance_x,
        'dict_cycle_x_devents': dict_cycle_x_devents, 
        'logical_error_probability':  1-X_good_trials/num_instance_x,
        'logical_error_probability_per_qubit': 1-X_good_list/num_instance_x
        })











