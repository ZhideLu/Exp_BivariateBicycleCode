
from tools_for_pauli_plus import *

basis_type = str(sys.argv[1])
num_cycles = int(sys.argv[2])

#----------------------------------------------------------------------------------------------------------------------
## import the decoder
from functions_BB_code import error_decoding
from ldpc import bposd_decoder, mod2
import pickle
with open('../Set_decoder_para_18_6_3.pkl', 'rb') as f:
    Set_decoder_para_18_6_3 = pickle.load(f)

my_bp_method = "ms"
my_max_iter = 2000
my_osd_method = "osd_cs"
my_osd_order = 7
my_ms_scaling_factor = 0

channel_probsX = Set_decoder_para_18_6_3[f"channel_probsX_{num_cycles}"]
HX = Set_decoder_para_18_6_3[f"HX_{num_cycles}"]  ;
HdecX = Set_decoder_para_18_6_3[f"HdecX_{num_cycles}"] ;

channel_probsZ = Set_decoder_para_18_6_3[f"channel_probsZ_{num_cycles}"]
HZ = Set_decoder_para_18_6_3[f"HZ_{num_cycles}"]  ;
HdecZ = Set_decoder_para_18_6_3[f"HdecZ_{num_cycles}"] ;

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
    HdecZ, 
    channel_probs=channel_probsZ, 
    max_iter=my_max_iter, 
    bp_method=my_bp_method,
    ms_scaling_factor=my_ms_scaling_factor, 
    osd_method="osd_cs", 
    osd_order=my_osd_order 
    )


# Error decoding
if basis_type == "Z":
    
    fname = './Numerical_data/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '.mat'
    data = loadmat(fname)
    
    retained_num_samples = data["retained_num_samples"][0][0] ;
    syndrome_history_Z = data["syndrome_history_Z"] ;
    final_logical_z_outcome = data["final_logical_z_outcome"] ;


    # correct errors for logical Z (Z checks)
    Z_no_correction_good_trials, Z_no_correction_good_list, Z_good_trials, Z_good_list =     \
                error_decoding(retained_num_samples, k, bpdZ, HZ, syndrome_history_Z, final_logical_z_outcome )

    print("error correction for logical Z basis:")
    print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-Z_good_trials/retained_num_samples)
    print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-Z_good_list/retained_num_samples)

    sname = './Numerical_results/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '.mat'
    savemat(sname, 
            {'num_cycles': num_cycles, 
            'logical_error_probability':  1-Z_good_trials/retained_num_samples,
            'logical_error_probability_per_qubit': 1-Z_good_list/retained_num_samples  })
#---------------------------------------------------------------------------------------------------------------------------------

if basis_type == "X":

    fname = './Numerical_data/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '.mat'
    data = loadmat(fname)
    
    retained_num_samples = data["retained_num_samples"][0][0] ;
    syndrome_history_X = data["syndrome_history_X"] ;
    final_logical_x_outcome = data["final_logical_x_outcome"] ;
    
    # correct errors for logical X (X checks)
    X_no_correction_good_trials, X_no_correction_good_list, X_good_trials, X_good_list =     \
                error_decoding(retained_num_samples, k, bpdX, HX, syndrome_history_X, final_logical_x_outcome )
    
    print("error correction for logical X basis:")
    print(f'Logical error over {num_cycles+1} cycles (four logical qubits):', 1-X_good_trials/retained_num_samples)
    print(f'Logical error over {num_cycles+1} cycles (single logical qubit):', 1-X_good_list/retained_num_samples)

    sname = './Numerical_results/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '.mat'
    savemat(sname, 
            {'num_cycles': num_cycles, 
            'logical_error_probability':  1-X_good_trials/retained_num_samples,
            'logical_error_probability_per_qubit': 1-X_good_list/retained_num_samples  })
#----------------------------------------------------------------------------------------------------------------------------------











