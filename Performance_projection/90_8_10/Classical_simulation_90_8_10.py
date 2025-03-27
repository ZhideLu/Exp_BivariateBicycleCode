import numpy as np
from scipy.sparse import coo_matrix, hstack 
from mip import Model, xsum, minimize, BINARY, OptimizationStatus
from bposd.css import css_code
from ldpc import bposd_decoder, mod2
import pickle
import itertools

import sys
import os

from BB_code_90_8_10 import ell, m, n, n2, A, B, AT, BT, num_end 
from BB_code_90_8_10 import distance_test, rank2, get_connection_Tanner, get_layout
from BB_code_90_8_10 import get_SM_circuit_parallel, get_SM_circuit_separate
from BB_code_90_8_10 import simulate_circuit_Lx, simulate_circuit_Lz
from BB_code_90_8_10 import get_Set_noisy_circuits_logical_X, get_Set_noisy_circuits_logical_Z
from BB_code_90_8_10 import decoding_X_matrix, decoding_Z_matrix
from BB_code_90_8_10 import simulate_syndrome_Lx, simulate_syndrome_Lz 
from BB_code_90_8_10 import generate_noisy_circuit
from BB_code_90_8_10 import error_decoding

from scipy.io import savemat
from scipy.io import loadmat

num_cycles = 6 ; 

##------------------------------------------------------------------------------------------------------------
# Code construction

# code [[90,8,10]]
ell,m = 3,15
a1,a2,a3 = 0,1,5
b1,b2,b3 = 3,1,2

d = 10 ;

# code length
n = 2*m*ell;  n2 = m*ell

I_ell = np.identity(ell,dtype=int) ; I_m = np.identity(m,dtype=int); I = np.identity(ell*m,dtype=int)
x = {} ;  y = {}

for i in range(ell):
	x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
for i in range(m):
	y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))

A = (x[a1%ell] + y[a2%m] + y[a3%m]) % 2
B = (y[b1%m] + x[b2%ell] + x[b3%ell]) % 2

A1 = x[a1%ell]; A2 = y[a2%m]; A3 = y[a3%m]
B1 = y[b1%m]; B2 = x[b2%ell]; B3 = x[b3%ell]

AT = np.transpose(A) ; BT = np.transpose(B)

# Testing CSS code
hx = np.hstack((A,B));  hz = np.hstack((BT,AT))

remove_X_list = [] ;  remove_Z_list = [] ;

hx = np.delete(hx, remove_X_list, axis=0) ; hz = np.delete(hz, remove_Z_list, axis=0)

# number of logical qubits
k = n - rank2(hx) - rank2(hz)

qcode=css_code(hx=hx, hz=hz)
# print('Testing CSS code...')
# qcode.test()

# logical operator
lz = qcode.lz ;  lx = qcode.lx

# the order of two-qubit gates
sX = ['idle', 'idle', 1, 4, 3, 5, 0, 2] ; sZ = ['idle', 3, 5, 0, 1, 2, 4, 'idle'] ;
##------------------------------------------------------------------------------------------------------------

# Connections of edges in the Tanner graph
lin_order, data_qubits, Xchecks, Zchecks, nbs = get_connection_Tanner(remove_X_list, remove_Z_list, n2)

SM_cycle = get_SM_circuit_parallel(remove_X_list, remove_Z_list, lin_order, data_qubits, Xchecks, Zchecks, nbs, sX, sZ ) ;

cycle_append = []
for q in data_qubits:
    cycle_append.append(('final',q))
##------------------------------------------------------------------------------------------------------------

scaling_factor = float(sys.argv[1])

# # depolarizing noise model 
sys.path.append(os.path.abspath(os.path.join("..", "..")))
#  depolarizing noise model 
from component_error_rates import error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip


error_rate_init *= scaling_factor
error_rate_idle *= scaling_factor
error_rate_H *= scaling_factor
error_rate_cz *= scaling_factor
error_rate_meas *= scaling_factor
error_final *= scaling_factor
error_DD_phaseflip *= scaling_factor
error_DD_bitflip *= scaling_factor


# classical simulation
##------------------------------------------------------------------------------------------------------
 
# logical X data from classical simulation
num_trials_x = 40000 ;
syndrome_history_X, final_logical_x_outcome = simulate_syndrome_Lx(num_trials_x, num_cycles, SM_cycle, cycle_append, error_rate_init, \
                error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip, lx, \
                                                              lin_order, data_qubits, Xchecks, remove_X_list, remove_Z_list)

assert( len(syndrome_history_X[0]) == (num_cycles+1) * (n2 - len(remove_X_list) ) )
num_detect_x = len(syndrome_history_X[0]) ;

fname1 = './Numerical_data/' + 'Logical_X_' + 'num_cycles_' + str(num_cycles) + '_scaling_' + str(scaling_factor) + '.mat'
savemat(fname1, 
        { 'num_cycles': num_cycles, 
         'num_trials_x': num_trials_x,
        'syndrome_history_X': syndrome_history_X, 
        'final_logical_x_outcome': final_logical_x_outcome
        })

##------------------------------------------------------------------------------------------------------
# logical Z data from classical simulation
num_trials_z = 40000 ;
syndrome_history_Z, final_logical_z_outcome = simulate_syndrome_Lz(num_trials_z, num_cycles, SM_cycle, cycle_append, error_rate_init, \
                error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip, lz, \
                                                        lin_order, data_qubits, Zchecks, remove_X_list, remove_Z_list)

assert( len(syndrome_history_Z[0]) == (num_cycles+1) * (n2 - len(remove_Z_list) ) )
num_detect_z = len(syndrome_history_Z[0]) ;

fname2 = './Numerical_data/' + 'Logical_Z_' + 'num_cycles_' + str(num_cycles) + '_scaling_' + str(scaling_factor) + '.mat'
savemat(fname2, 
        { 'num_cycles': num_cycles, 
         'num_trials_z': num_trials_z,
        'syndrome_history_Z': syndrome_history_Z, 
        'final_logical_z_outcome': final_logical_z_outcome
        })













