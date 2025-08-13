from typing import List, Sequence, Tuple
import itertools
import functools
import numpy as np

from collections import defaultdict
from itertools import permutations
from tabulate import tabulate

from scipy.io import savemat
from scipy.io import loadmat

import stim
from bposd.css import css_code

from qutip import *
from qutip import to_choi

import os
import sys
sys.path.append(os.path.abspath("../../"))
from functions_BB_code import distance_test, rank2, get_connection_Tanner, get_layout

# from crosstalk_data import *
from packages import *

num_level = 4 ;

##-----------------------------------------------------------------------------------------------------------------------------------------
def f2(x): return 0 if x in [0, 1] else x-1

##-----------------------------------------------------------------------------------------------------------------------------------------
# Decay, dephasing, and leakage heating
def generate_channel_t(t_gate):
    dim = 4
    a = destroy(dim)
    n = a.dag() * a
    
    T1 = 41.8 ; 
    Tphi = 39.7 ;
    Gamma_12 = 1 / 1200 ;
    
    c_ops = [
        np.sqrt(1/T1) * a,
        np.sqrt(2/Tphi) * n,
        np.sqrt(Gamma_12) * basis(dim,2) * basis(dim,1).dag()
    ] ;
    
    L = liouvillian(0 * qeye(dim), c_ops) ;
    E_super = (L * t_gate).expm() ;
    choi = to_choi(E_super)  ;
    
    eigvals, eigvecs = choi.eigenstates() ;
    kraus_ops = [] ;
    for k in range(len(eigvals)):
        if eigvals[k] > 1e-15:  # 过滤小特征值
            d = dim
            vec = eigvecs[k].full().reshape((d, d), order = "F")
            K = np.sqrt(eigvals[k]) * Qobj(vec)
            kraus_ops.append(K)
    
    # sum([K.dag() * K for K in kraus_ops])
    kraus_cz = [_.full() for _ in kraus_ops]
    channel = kraus_to_leaky_pauli_channel(kraus_cz, 1, 4)
    return channel

# CZ leakage
def generate_channel_cz(leak_transit):
    dict_leak_transit = {} ;
    for i in range(num_level**2):
        for j in range(num_level**2):
            key = f"[{f2(i//4)}{f2(i%4)},{f2(j//4)}{f2(j%4)}]"
            dict_leak_transit.setdefault(key, []).append(leak_transit[i,j])
    ##-------------------------------------------------------------------------------------------------------------------
    dict_leak_transit = {
        k: np.sum(v) / ( 2 ** k.strip('[]').split(',')[0].count('0') )
        for k, v in dict_leak_transit.items()
        if np.sum(v) != 0
    }
    ##-------------------------------------------------------------------------------------------------------------------
    sum_by_source = defaultdict(float)
    for k, v in dict_leak_transit.items():
        src = k.strip('[]').split(',')[0]
        sum_by_source[src] += v
    
    dict_leak_transit = {
        k: v / sum_by_source[k.strip('[]').split(',')[0]]
        for k, v in dict_leak_transit.items()
    }
    
    channel_cz = leaky.LeakyPauliChannel(is_single_qubit_channel=False)
    for k, v in dict_leak_transit.items():
        src, tgt = k.strip('[]').split(',')
        k1 = int(src[0]) * num_level**2 + int(src[1])
        k2 = int(tgt[0]) * num_level**2 + int(tgt[1])
        channel_cz.add_transition(k1, k2, 0, v) 
    return channel_cz
##-----------------------------------------------------------------------------------------------------------------------------------------
# Three-state measurement errors

three_state_level = [0, 1, 2] ;
check_three_meas_error = np.array([[0.94265103, 0.02431824, 0.03303073],
                                   [0.0491197 , 0.91682406, 0.03405623],
                                   [0.04242471, 0.05032428, 0.907251  ],
                                   [0.33333333, 0.33333333, 0.33333333]]) ;

data_three_meas_error = np.array([[0.98202796, 0.01202925, 0.00594279],
                                  [0.04854744, 0.91835419, 0.03309837],
                                  [0.0143475 , 0.05631824, 0.92933426],
                                  [0.33333333, 0.33333333, 0.33333333]]) ;

## adding readout error in postprocessing
def apply_transition(arr, trans_mat):
    arr = np.asarray(arr)
    new_arr = np.empty_like(arr)
    unique_states = np.unique(arr)
    
    for state in unique_states:
        mask = (arr == state)  # 布尔掩码矩阵
        num_samples = np.sum(mask)

        sampled = np.random.choice(three_state_level, size = num_samples, p = trans_mat[state])

        new_arr[mask] = sampled
    return new_arr

def apply_transition_2d(num_databits, arr, trans_mat_1, trans_mat_2):
    arr = np.asarray(arr)
    new_arr = np.empty_like(arr)

    new_arr[:, :-num_databits] = apply_transition(arr[:, :-num_databits], trans_mat_1)

    new_arr[:, -num_databits:] = apply_transition(arr[:, -num_databits:], trans_mat_2)

    return new_arr
##-----------------------------------------------------------------------------------------------------------------------------------------
# circuit represented with Stim
# BB code [[18,4,4]]
ell,m = 3, 3 ; a1,a2,a3 = 1, 0, 2 ; b1,b2,b3 = 1, 0, 2

# code length
n = 2*m*ell;  n2 = m*ell ; remove_X_list = [2,8] ;  remove_Z_list = [3,4] ;

I_ell = np.identity(ell,dtype=int) ; I_m = np.identity(m,dtype=int); I = np.identity(ell*m,dtype=int)
x = {} ;  y = {}

for i in range(ell):
	x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
for i in range(m):
	y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))

A = (x[a1%ell] + y[a2%m] + y[a3%m]) % 2;  B = (y[b1%m] + x[b2%ell] + x[b3%ell]) % 2
A1 = x[a1%ell]; A2 = y[a2%m]; A3 = y[a3%m] ; B1 = y[b1%m]; B2 = x[b2%ell]; B3 = x[b3%ell]
AT = np.transpose(A) ; BT = np.transpose(B) ;

# Testing CSS code
hx = np.hstack((A,B));  hz = np.hstack((BT,AT))
hx = np.delete(hx, remove_X_list, axis=0) ; hz = np.delete(hz, remove_Z_list, axis=0)

# number of logical qubits
k = n - rank2(hx) - rank2(hz)
qcode=css_code(hx=hx, hz=hz)

# logical operator
lz = qcode.lz ;  lx = qcode.lx
# we choose a set of basis such that each logical X anticommute with its corresponding logical Z.
lz[1] = (lz[1]+lz[2]) %2 ; lz[3] = (lz[3]+lz[0]) %2

lz_copy = lz.copy()
lz_copy[0] = lz[1] ;  lz_copy[1] = lz[0] ;  lz_copy[2] = lz[3] ;  lz_copy[3] = lz[2] ;
lz = lz_copy
# (lx@lz.T) %2
#----------------------------------------------------------------------------------------------------------------------
sX = ['idle', 'idle', 1, 4, 3, 5, 0, 2] ; sZ = ['idle', 3, 5, 0, 1, 2, 4, 'idle'] ;
# Connections of edges in the Tanner graph
lin_order, data_qubits, Xchecks, Zchecks, nbs = get_connection_Tanner(remove_X_list, remove_Z_list, n2)
#----------------------------------------------------------------------------------------------------------------------
Xcheck = np.arange(n2 - len(remove_X_list) ).tolist()
data_L = np.arange(n2 - len(remove_X_list), 2 * n2 - len(remove_X_list)).tolist()
data_R = np.arange(2 * n2 - len(remove_X_list), 3 * n2 - len(remove_X_list)).tolist()
Zcheck = np.arange(3 * n2 - len(remove_X_list), 4 * n2 - len(remove_X_list) - len(remove_Z_list) ).tolist()

num_qubits = 4 * n2 - len(remove_X_list) - len(remove_Z_list) # 32 physical qubits
num_check_Z = n2 - len(remove_Z_list) ; num_check_X = n2 - len(remove_X_list) ; num_databits = len(data_L + data_R) ;
num_checks = num_qubits - num_databits ;
##----------------------------------------------------------------------------------------------------------------------

# SQRT_Y: bind channel_sq for isolated H gates
# SQRT_X: bind channel_DD for DD 
# S: bind channel_tq for CZ gates
# I: bind channel_idle for qubit errors for iding during CZ 
# CZ: bind channel_cz for U_{leakage} 

##----------------------------------------------------------------------------------------------------------------------
set_higher_fre_Z = [(('Zcheck', 5),('data_right', 2)), (('Zcheck', 1),('data_right', 2)), (('Zcheck', 8),('data_left', 7)),
                   (('Zcheck', 7),('data_left', 6)), (('Zcheck', 5),('data_left', 4)), (('Zcheck', 7),('data_left', 7)),
                   (('Zcheck', 6),('data_left', 6)), (('Zcheck', 1),('data_left', 4)), (('Zcheck', 2),('data_right', 2))  ] ;

set_higher_fre_X = [(('Xcheck', 7),('data_left', 7)), (('Xcheck', 4),('data_left', 4)), (('Xcheck', 6),('data_left', 6)),
                   (('Xcheck', 1),('data_right', 2)), (('Xcheck', 5),('data_right', 2)), (('Xcheck', 3),('data_left', 6)),
                   (('Xcheck', 4),('data_left', 7)), (('Xcheck', 1),('data_left', 4)), (('Xcheck', 7),('data_left', 6)),
                   (('Xcheck', 5),('data_left', 4))  ] ;
##----------------------------------------------------------------------------------------------------------------------

def get_SM_circuit(error_rate_sq_com, error_rate_cz_com, error_rate_idle_com, error_crosstalk, remove_X_list, remove_Z_list, lin_order, data_qubits, Xchecks, Zchecks, nbs, sX, sZ):
    circuit = stim.Circuit()
    circuit.append("H", Xcheck + data_L + Zcheck)
    circuit.append("SQRT_Y_DAG", Xcheck + data_L + Zcheck)
    circuit.append("SQRT_Y", Xcheck + data_L + Zcheck)  
    circuit.append("DEPOLARIZE1", Xcheck + data_L + Zcheck, error_rate_sq_com)
##----------------------------------------------------------------------------------------------------------------------
    t = 1
    data_qubits_cnoted_in_this_round = []
    assert(not(sZ[t]=='idle'))
    for target in Zchecks:
        direction = sZ[t]
        control = nbs[(target,direction)]
        data_qubits_cnoted_in_this_round.append(control)
        
        if (target, control) in set_higher_fre_Z:
            circuit.append("CZ", [lin_order[target], lin_order[control] ] )
        else:
            circuit.append("CZ", [lin_order[control], lin_order[target]] )

        circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
        circuit.append("S", [lin_order[target], lin_order[control] ] )
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            circuit.append("I", [lin_order[q]] )
            circuit.append("DEPOLARIZE1", [lin_order[q]], error_rate_idle_com)

    circuit.append("I", Xcheck)
    circuit.append("DEPOLARIZE1", Xcheck, error_rate_idle_com)
    circuit.append("Y", Xcheck + data_L + data_R + Zcheck)   #
    circuit.append("DEPOLARIZE1", Xcheck + data_L + data_R + Zcheck, error_rate_sq_com)
##----------------------------------------------------------------------------------------------------------------------
    # for ck in order_crosstalk_gpc_SQ[t]:
    #     circuit.append("SQRT_XX_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
    #     circuit.append("SQRT_XX", [ lin_order[ck[0]], lin_order[ck[1]] ])   
##----------------------------------------------------------------------------------------------------------------------
    t = 2
    data_qubits_cnoted_in_this_round = []
    
    assert(not(sX[t]=='idle'))
    for control in Xchecks:
        direction = sX[t]
        target = nbs[(control,direction)]
        data_qubits_cnoted_in_this_round.append(target)   

        if (control, target) in set_higher_fre_X:
            circuit.append("CZ",[lin_order[control], lin_order[target]] )
        else:
            circuit.append("CZ", [lin_order[target], lin_order[control]] )

        circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
        circuit.append("S", [lin_order[target], lin_order[control] ] )
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
    assert(not(sZ[t]=='idle'))
    for target in Zchecks:
        direction = sZ[t]
        control = nbs[(target,direction)] ;
        data_qubits_cnoted_in_this_round.append(control) 
        
        if (target, control) in set_higher_fre_Z:
            circuit.append("CZ", [lin_order[target], lin_order[control] ] )
        else:
            circuit.append("CZ", [lin_order[control], lin_order[target] ] )

        circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
        circuit.append("S", [lin_order[target], lin_order[control] ] )
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            circuit.append("I", [lin_order[q]] )
            circuit.append("DEPOLARIZE1", [lin_order[q]], error_rate_idle_com)
    circuit.append("Y", Xcheck + data_L + data_R + Zcheck)  #

    for q in data_qubits:  
        circuit.append("H", [lin_order[q]])
    circuit.append("DEPOLARIZE1", Xcheck + data_L + data_R + Zcheck, error_rate_sq_com)
##----------------------------------------------------------------------------------------------------------------------
    # for ck in order_crosstalk_gpc[t]:
    #     circuit.append("ISWAP_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
    #     circuit.append("ISWAP", [ lin_order[ck[0]], lin_order[ck[1]] ])
    # for ck in order_crosstalk_gpc_SQ[t]:
    #     circuit.append("SQRT_XX_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
    #     circuit.append("SQRT_XX", [ lin_order[ck[0]], lin_order[ck[1]] ])      
##-------------------------------------------------------------------------------------------------------------
    for t in range(3,6):
        data_qubits_cnoted_in_this_round = []
        
        assert(not(sX[t]=='idle'))
        for control in Xchecks:
            direction = sX[t]
            target = nbs[(control,direction)]
            data_qubits_cnoted_in_this_round.append(target)   
    
            if (control, target) in set_higher_fre_X:
                circuit.append("CZ",[lin_order[control], lin_order[target]] )
            else:
                circuit.append("CZ", [lin_order[target], lin_order[control]] )

            circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
            circuit.append("S", [lin_order[target], lin_order[control] ] )
            circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
            circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
        assert(not(sZ[t]=='idle'))
        for target in Zchecks:
            direction = sZ[t]
            control = nbs[(target,direction)] ;
            data_qubits_cnoted_in_this_round.append(control) 

            if (target, control) in set_higher_fre_Z:
                circuit.append("CZ", [lin_order[target], lin_order[control] ] )
            else:
                circuit.append("CZ", [lin_order[control], lin_order[target]] )
            
            circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
            circuit.append("S", [lin_order[target], lin_order[control] ] )
            circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
            circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
        for q in data_qubits:
            if not(q in data_qubits_cnoted_in_this_round):
                circuit.append("I", [lin_order[q]] )
                circuit.append("DEPOLARIZE1", [lin_order[q]], error_rate_idle_com)
        circuit.append("Y", Xcheck + data_L + data_R + Zcheck)  #
        circuit.append("DEPOLARIZE1", Xcheck + data_L + data_R + Zcheck, error_rate_sq_com)
##----------------------------------------------------------------------------------------------------------------------
        # for ck in order_crosstalk_gpc[t]:
        #     circuit.append("ISWAP_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
        #     circuit.append("ISWAP", [ lin_order[ck[0]], lin_order[ck[1]] ])
        # for ck in order_crosstalk_gpc_SQ[t]:
        #     circuit.append("SQRT_XX_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
        #     circuit.append("SQRT_XX", [ lin_order[ck[0]], lin_order[ck[1]] ])   
##----------------------------------------------------------------------------------------------------------------------
    for q in data_qubits:
        circuit.append("H", [lin_order[q]])
##-------------------------------------------------------------------------------------------------------------
    t = 6
    data_qubits_cnoted_in_this_round = []
    
    assert(not(sX[t]=='idle'))
    for control in Xchecks:
        direction = sX[t]
        target = nbs[(control,direction)]
        data_qubits_cnoted_in_this_round.append(target)   

        if (control, target) in set_higher_fre_X:
            circuit.append("CZ",[lin_order[control], lin_order[target]] )
        else:
            circuit.append("CZ", [lin_order[target], lin_order[control]] )

        circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
        circuit.append("S", [lin_order[target], lin_order[control] ] )
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
    assert(not(sZ[t]=='idle'))
    for target in Zchecks:
        direction = sZ[t]
        control = nbs[(target,direction)] ;
        data_qubits_cnoted_in_this_round.append(control) 
        
        if (target, control) in set_higher_fre_Z:
            circuit.append("CZ", [lin_order[target], lin_order[control] ] )
        else:
            circuit.append("CZ", [lin_order[control], lin_order[target] ] )

        circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
        circuit.append("S", [lin_order[target], lin_order[control] ] )
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            circuit.append("I", [lin_order[q]] )
            circuit.append("DEPOLARIZE1", [lin_order[q]], error_rate_idle_com)
    circuit.append("Y", Xcheck + data_L + data_R + Zcheck)  #
    circuit.append("DEPOLARIZE1", Xcheck + data_L + data_R + Zcheck, error_rate_sq_com)
##----------------------------------------------------------------------------------------------------------------------
    # for ck in order_crosstalk_gpc[t]:
    #     circuit.append("ISWAP_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
    #     circuit.append("ISWAP", [ lin_order[ck[0]], lin_order[ck[1]] ])
    # for ck in order_crosstalk_gpc_SQ[t]:
    #     circuit.append("SQRT_XX_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
    #     circuit.append("SQRT_XX", [ lin_order[ck[0]], lin_order[ck[1]] ])   
##-------------------------------------------------------------------------------------------------------------
    t = 7
    assert(not(sX[t]=='idle'))
    data_qubits_cnoted_in_this_round = []
    for control in Xchecks:
        direction = sX[t]
        target = nbs[(control,direction)]

        if (control, target) in set_higher_fre_X:
            circuit.append("CZ",[lin_order[control], lin_order[target]] )
        else:
            circuit.append("CZ", [lin_order[target], lin_order[control]] )

        circuit.append("S_DAG", [lin_order[target], lin_order[control] ] )
        circuit.append("S", [lin_order[target], lin_order[control] ] )
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_rate_cz_com)
        circuit.append("DEPOLARIZE2", [lin_order[target], lin_order[control]], error_crosstalk)
        data_qubits_cnoted_in_this_round.append(target)
        
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            circuit.append("I", [lin_order[q]] )
            circuit.append("DEPOLARIZE1", [lin_order[q]], error_rate_idle_com)
    circuit.append("I", Zcheck)
    circuit.append("DEPOLARIZE1", Zcheck, error_rate_idle_com)
##----------------------------------------------------------------------------------------------------------------------
    # for ck in order_crosstalk_gpc[t]:
    #     circuit.append("ISWAP_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
    #     circuit.append("ISWAP", [ lin_order[ck[0]], lin_order[ck[1]] ])
    # for ck in order_crosstalk_gpc_SQ[t]:
    #     circuit.append("SQRT_XX_DAG", [ lin_order[ck[0]], lin_order[ck[1]] ])
    #     circuit.append("SQRT_XX", [ lin_order[ck[0]], lin_order[ck[1]] ])   
##----------------------------------------------------------------------------------------------------------------------
    circuit.append("Z", [x for x in Xcheck + Zcheck if (not x in [4,5,27,28]) ] )
    
    circuit.append("X", [7,8,9,13,14, 17,18,20,23,24] )
    circuit.append("Y", [12,15, 19,21] )

    circuit.append("H", Xcheck + data_L + Zcheck)

    circuit.append("SQRT_Y_DAG", [4,5, 10,11, 27,28] )
    circuit.append("SQRT_Y", [4,5, 10,11, 27,28]  )
    circuit.append("DEPOLARIZE1", [p for p in Xcheck + data_L + data_R + Zcheck if p not in [16, 22]], error_rate_sq_com)
##----------------------------------------------------------------------------------------------------------------------
    circuit.append("M", Xcheck + Zcheck)
    return circuit
##--------------------------------------------------------------------------------------------------------------------------------


# Converting measurement outcomes into error detections¶
# only reserve Z check
def Z_transfer_meas_to_detection(hz, lz, num_cycles, results, num_databits):

    num_trials_z = results.shape[0] ;
    syndrome_history_Z = [] ;
    final_logical_z_outcome = [] ;
#----------------------------------------------------------------------------------------------------------------------
    for i in range(num_trials_z):
        data_meas = results[i][-num_databits:] ;
        syndrome_final_logical = (lz @ data_meas.T) % 2  
        final_logical_z_outcome.append(syndrome_final_logical.tolist() )
#----------------------------------------------------------------------------------------------------------------------
        Z_check_meas = [results[i][:-num_databits][j] for block in range(num_cycles) 
                          for j in range(block*num_checks + num_check_Z, (block+1)*num_checks)];
        assert(  len(Z_check_meas) == num_cycles * num_check_Z )
        syndrome_history = Z_check_meas + ((hz @ data_meas.T) % 2).tolist()
        syndrome_history = np.array(syndrome_history)
#----------------------------------------------------------------------------------------------------------------------
        syndrome_history_copy = syndrome_history.copy()
        # measurement_history ---> detection events
        if num_cycles == 1:
            assert(len(syndrome_history) == (num_cycles + 1)*num_check_Z)
            syndrome_history[num_cycles*num_check_Z : ] += syndrome_history_copy[ : num_cycles*num_check_Z] ;
    
        if num_cycles > 1:
            assert(len(syndrome_history) == (num_cycles + 1)*num_check_Z)
            syndrome_history[-1*num_check_Z : ] += ( syndrome_history[-2*num_check_Z : -1*num_check_Z] \
                        + syndrome_history[-3*num_check_Z : -2*num_check_Z] )

            for row in range(2, num_cycles):
                syndrome_history[row*num_check_Z : (row+1)*num_check_Z] += \
                                syndrome_history_copy[(row-2)*num_check_Z : (row-1)*num_check_Z]  ;
        syndrome_history%= 2  
        syndrome_history_Z.append( syndrome_history.tolist() )
    return np.array(syndrome_history_Z), np.array(final_logical_z_outcome)
#-------------------------------------------------------------------------------------------------------------------------------

# only reserve X check
def X_transfer_meas_to_detection(hx, lx, num_cycles, results, num_databits):

    num_trials_x = results.shape[0] ;
    syndrome_history_X = [] ;
    final_logical_x_outcome = [] ;
#----------------------------------------------------------------------------------------------------------------------
    for i in range(num_trials_x):
        data_meas = results[i][-num_databits:] ;
        syndrome_final_logical = (lx @ data_meas.T) % 2  
        final_logical_x_outcome.append(syndrome_final_logical.tolist() )
#----------------------------------------------------------------------------------------------------------------------
        X_check_meas = [results[i][:-num_databits][j] for block in range(num_cycles) 
                          for j in range(block*num_checks, block*num_checks + num_check_X)];
        assert(  len(X_check_meas) == num_cycles * num_check_X )
        syndrome_history = X_check_meas + ((hx @ data_meas.T) % 2).tolist()
        syndrome_history = np.array(syndrome_history)
#----------------------------------------------------------------------------------------------------------------------
        syndrome_history_copy = syndrome_history.copy()
        # measurement_history ---> detection events
        if num_cycles == 1:
            assert(len(syndrome_history) == (num_cycles + 1)*num_check_X)
            syndrome_history[num_cycles*num_check_X : ] += syndrome_history_copy[ : num_cycles*num_check_X] ;
    
        if num_cycles > 1:
            assert(len(syndrome_history) == (num_cycles + 1)*num_check_X)
            syndrome_history[-1*num_check_X : ] += ( syndrome_history[-2*num_check_X : -1*num_check_X] \
                        + syndrome_history[-3*num_check_X : -2*num_check_X] )

            for row in range(2, num_cycles):
                syndrome_history[row*num_check_X : (row+1)*num_check_X] += \
                                syndrome_history_copy[(row-2)*num_check_X : (row-1)*num_check_X]  ;
        syndrome_history%= 2  
        syndrome_history_X.append( syndrome_history.tolist() )
    return np.array(syndrome_history_X), np.array(final_logical_x_outcome)














