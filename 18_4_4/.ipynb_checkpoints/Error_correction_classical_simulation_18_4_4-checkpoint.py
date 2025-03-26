import numpy as np
from mip import Model, xsum, minimize, BINARY
from mip import OptimizationStatus
from bposd.css import css_code
from ldpc import bposd_decoder
import pickle
import itertools
from ldpc import bposd_decoder 
from scipy.sparse import coo_matrix, hstack 
from ldpc import mod2

from scipy.io import savemat
from scipy.io import loadmat

import sys
import os
sys.path.append(os.path.abspath("..")) 
from functions_BB_code import error_decoding

##------------------------------------------------------------------------------------------------------------
# Code construction
num_L = 9 ; num_R = 9 ;
k = 4 ; n2 = 9 ;
index_Z_check = [0, 1, 2, 5, 6, 7, 8] ;
index_X_check = [0, 1, 3, 4, 5, 6, 7] ;

remove_X_list = [2,8] ;  remove_Z_list = [3,4] ;

with open('Set_decoder_para_18_4_4.pkl', 'rb') as f:
    Set_decoder_para_18_4_4 = pickle.load(f)
    
# Check matrix and logical operators for the BB code [[18,4,4]] 
hz = Set_decoder_para_18_4_4["hz"] ;
lz = Set_decoder_para_18_4_4["lz"] ;

hx = Set_decoder_para_18_4_4["hx"] ;
lx = Set_decoder_para_18_4_4["lx"] ;

# classical simulation
##------------------------------------------------------------------
num_cycles = int(sys.argv[1])
##------------------------------------------------------------------
## import the decoder
##------------------------------------------------------------------
my_bp_method = "ms"
my_max_iter = 10000
my_osd_method = "osd_cs"
my_osd_order = 7
my_ms_scaling_factor = 0

channel_probsX = Set_decoder_para_18_4_4[f"channel_probsX_{num_cycles}"]
HX = Set_decoder_para_18_4_4[f"HX_{num_cycles}"]  ;
HdecX = Set_decoder_para_18_4_4[f"HdecX_{num_cycles}"] ;

channel_probsZ = Set_decoder_para_18_4_4[f"channel_probsZ_{num_cycles}"]
HZ = Set_decoder_para_18_4_4[f"HZ_{num_cycles}"]  ;
HdecZ = Set_decoder_para_18_4_4[f"HdecZ_{num_cycles}"] ;

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
##-------------------------------------------------------------------------------------------------------------------------

# logical Z
# data from classical simulation¶

fname_z = './Numerical_data/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '.mat'
data_z = loadmat(fname_z)

num_trials_z = data_z["num_trials_z"][0][0] ;
syndrome_history_Z = data_z["syndrome_history_Z"] ;
final_logical_z_outcome = data_z["final_logical_z_outcome"] ;


# correct errors for logical Z (Z checks)
Z_no_correction_good_trials, Z_no_correction_good_list, Z_good_trials, Z_good_list =     \
            error_decoding(num_trials_z, k, bpdZ, HZ, syndrome_history_Z, final_logical_z_outcome )


print("no error correction for logical Z basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-Z_no_correction_good_trials/num_trials_z)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-Z_no_correction_good_list/num_trials_z)
print("\n")
print("error correction for logical Z basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-Z_good_trials/num_trials_z)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-Z_good_list/num_trials_z)


fname1 = './Numerical_results/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '.mat'
savemat(fname1, 
        { 'num_cycles': num_cycles, 
        'syndrome_history_Z': syndrome_history_Z, 
        'logical_error_probability':  1-Z_good_trials/num_trials_z,
        'logical_error_probability_per_qubit': 1-Z_good_list/num_trials_z
        })

##-----------------------------------------------------------------------------------------------------------------------

# logical X
# data from classical simulation¶
fname_x = './Numerical_data/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '.mat'
data_x = loadmat(fname_x)

num_trials_x = data_x["num_trials_x"][0][0] ;
syndrome_history_X = data_x["syndrome_history_X"] ;
final_logical_x_outcome = data_x["final_logical_x_outcome"] ;


# correct errors for logical X (X checks)
X_no_correction_good_trials, X_no_correction_good_list, X_good_trials, X_good_list =     \
            error_decoding(num_trials_x, k, bpdX, HX, syndrome_history_X, final_logical_x_outcome )


print("no error correction for logical X basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-X_no_correction_good_trials/num_trials_x)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-X_no_correction_good_list/num_trials_x)
print("\n")
print("error correction for logical X basis:")
print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-X_good_trials/num_trials_x)
print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-X_good_list/num_trials_x)


fname2 = './Numerical_results/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '.mat'
savemat(fname2, 
        { 'num_cycles': num_cycles, 
        'syndrome_history_X': syndrome_history_X, 
        'logical_error_probability':  1-X_good_trials/num_trials_x,
        'logical_error_probability_per_qubit': 1-X_good_list/num_trials_x
        })


























