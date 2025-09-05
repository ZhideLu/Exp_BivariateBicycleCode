import numpy as np
from mip import Model, xsum, minimize, BINARY
from mip import OptimizationStatus
from bposd.css import css_code
import pickle
import itertools
from ldpc import bposd_decoder
from scipy.sparse import coo_matrix, hstack 
from ldpc import mod2

from scipy.io import savemat
from scipy.io import loadmat

import sys
import os

from BB_code_36_4_6 import error_decoding
num_cycles = 6 ;

##------------------------------------------------------------------------------------------------------------
# Code construction

scaling_factor = float(sys.argv[1])

iter_collect = int(sys.argv[2])

num_per_run = 400;
##------------------------------------------------------------------------------------------------------------

k = 4 ; d = 6 ;

with open(f'Decoder/Set_decoder_para_36_4_6_scaling{scaling_factor}.pkl', 'rb') as f:
    Set_decoder_para = pickle.load(f)
    
# Check matrix and logical operators 
hz = Set_decoder_para["hz"] ;
lz = Set_decoder_para["lz"] ;

hx = Set_decoder_para["hx"] ;
lx = Set_decoder_para["lx"] ;

# classical simulation
##------------------------------------------------------------------
## import the decoder
##------------------------------------------------------------------
my_bp_method = "ms"
my_max_iter = 10000
my_osd_method = "osd_cs"
my_osd_order = 7
my_ms_scaling_factor = 0

channel_probsX = Set_decoder_para[f"channel_probsX_{num_cycles}"]
HX = Set_decoder_para[f"HX_{num_cycles}"]  ;
HdecX = Set_decoder_para[f"HdecX_{num_cycles}"] ;

channel_probsZ = Set_decoder_para[f"channel_probsZ_{num_cycles}"]
HZ = Set_decoder_para[f"HZ_{num_cycles}"]  ;
HdecZ = Set_decoder_para[f"HdecZ_{num_cycles}"] ;

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

fname_z = './Numerical_data/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '_scaling_' + str(scaling_factor) + '.mat'
data_z = loadmat(fname_z)

num_trials_z = num_per_run;
syndrome_history_Z = data_z["syndrome_history_Z"][ num_per_run*iter_collect:num_per_run*(iter_collect+1) ] ;
final_logical_z_outcome = data_z["final_logical_z_outcome"][ num_per_run*iter_collect:num_per_run*(iter_collect+1) ] ;


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


fname1 = './Numerical_results/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '_scaling_' + str(scaling_factor) + '_iterCollect_' + str(iter_collect) + '.mat'
savemat(fname1, 
        { 'num_cycles': num_cycles, 
        'syndrome_history_Z': syndrome_history_Z, 
        'logical_error_probability':  1-Z_good_trials/num_trials_z,
        'logical_error_probability_per_qubit': 1-Z_good_list/num_trials_z
        })

##-----------------------------------------------------------------------------------------------------------------------

# logical X
# data from classical simulation¶
fname_x = './Numerical_data/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '_scaling_' + str(scaling_factor) + '.mat'

data_x = loadmat(fname_x)

num_trials_x = num_per_run;
syndrome_history_X = data_x["syndrome_history_X"][ num_per_run*iter_collect:num_per_run*(iter_collect+1) ] ; 
final_logical_x_outcome = data_x["final_logical_x_outcome"][ num_per_run*iter_collect:num_per_run*(iter_collect+1) ] ; 


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


fname2 = './Numerical_results/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '_scaling_' + str(scaling_factor) +'_iterCollect_' + str(iter_collect) + '.mat'

savemat(fname2, 
        { 'num_cycles': num_cycles, 
        'syndrome_history_X': syndrome_history_X, 
        'logical_error_probability':  1-X_good_trials/num_trials_x,
        'logical_error_probability_per_qubit': 1-X_good_list/num_trials_x
        })


























