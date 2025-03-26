import numpy as np
from mip import Model, xsum, minimize, BINARY
from mip import OptimizationStatus
from bposd.css import css_code
from ldpc import bposd_decoder
import pickle
import itertools 
from scipy.sparse import coo_matrix, hstack 
from ldpc import mod2
from tabulate import tabulate
import matplotlib.pyplot as plt

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

num_cycles = 6 ;

scaling_factor = float(sys.argv[1])

# Ted's code [[90,8,10]]
ell,m = 15,3  
a1,a2,a3 = 9,1,2    
b1,b2,b3 = 0,2,7

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

# Connections of edges in the Tanner graph
lin_order, data_qubits, Xchecks, Zchecks, nbs = get_connection_Tanner(remove_X_list, remove_Z_list, n2)

SM_cycle = get_SM_circuit_parallel(remove_X_list, remove_Z_list, lin_order, data_qubits, Xchecks, Zchecks, nbs, sX, sZ ) ;

cycle_append = []
for q in data_qubits:
    cycle_append.append(('final',q))


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


# Setup BP-OSD decoder parameters
my_bp_method = "ms"
my_max_iter = 10000
my_osd_method = "osd_cs"
my_osd_order = 7
my_ms_scaling_factor = 0


Set_decoder_para = {} ;

# full syndrome measurement circuit
cycle_repeated = (num_cycles-1) * SM_cycle[2*n2-len(remove_X_list)-len(remove_Z_list):] + \
            SM_cycle[2*n2-len(remove_X_list)-len(remove_Z_list) : -2*n2] + cycle_append  ;

# Generating noisy circuits with a singe faulty operation
X_circuits, X_Prob = get_Set_noisy_circuits_logical_X(cycle_repeated, error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, \
                                                      error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip)
num_errX=len(X_circuits)
print('Number of noisy circuits for the logical X state =',num_errX)


Z_circuits, Z_Prob = get_Set_noisy_circuits_logical_Z(cycle_repeated, error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, \
                                                      error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip)
num_errZ=len(Z_circuits)
print('Number of noisy circuits for the logical Z state =',num_errZ)    

channel_probsX, HX, HdecX, HXdict = decoding_X_matrix(X_circuits, X_Prob, num_cycles, SM_cycle, lin_order, n, k, data_qubits, \
                                                      Xchecks, lx, remove_X_list, remove_Z_list) ;

channel_probsZ, HZ, HdecZ, HZdict = decoding_Z_matrix(Z_circuits, Z_Prob, num_cycles, SM_cycle, lin_order, n, k, data_qubits, \
                                                      Zchecks, lz, remove_X_list, remove_Z_list) ;

Set_decoder_para[f"channel_probsX_{num_cycles}"] = channel_probsX ;
Set_decoder_para[f"HX_{num_cycles}"] = HX ;
Set_decoder_para[f"HdecX_{num_cycles}"] = HdecX ;

Set_decoder_para[f"channel_probsZ_{num_cycles}"] = channel_probsZ ;
Set_decoder_para[f"HZ_{num_cycles}"] = HZ ;
Set_decoder_para[f"HdecZ_{num_cycles}"] = HdecZ ;  

Set_decoder_para["hx"] = hx ;
Set_decoder_para["hz"] = hz ;
Set_decoder_para["lx"] = lx ;
Set_decoder_para["lz"] = lz ;


with open(f'Decoder/Set_decoder_para_90_8_10_scaling{scaling_factor}.pkl', 'wb') as f:
    pickle.dump(Set_decoder_para, f)
















