import numpy as np
import random
from scipy.sparse import coo_matrix, hstack 
from ldpc import bposd_decoder, mod2
from mip import Model, xsum, minimize, BINARY, OptimizationStatus
import pickle
import itertools

def rank2(A):
    rows,n = A.shape
    X = np.identity(n,dtype=int)

    for i in range(rows):
        y = np.dot(A[i,:], X) % 2
        not_y = (y + 1) % 2
        good = X[:,np.nonzero(not_y)]
        good = good[:,0,:]
        bad = X[:, np.nonzero(y)]
        bad = bad[:,0,:]
        if bad.shape[1]>0 :
            bad = np.add(bad,  np.roll(bad, 1, axis=1) ) 
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]
#------------------------------------------------------------------------------------------------------------------

def distance_test(stab,logicOp):
	# number of qubits
	n = stab.shape[1]
	# number of stabilizers
	m = stab.shape[0]

	# maximum stabilizer weight
	wstab = np.max([np.sum(stab[i,:]) for i in range(m)])
	# weight of the logical operator
	wlog = np.count_nonzero(logicOp)
	# how many slack variables are needed to express orthogonality constraints modulo two
	num_anc_stab = int(np.ceil(np.log2(wstab)))
	num_anc_logical = int(np.ceil(np.log2(wlog)))
	# total number of variables
	num_var = n + m*num_anc_stab + num_anc_logical

	model = Model()
	model.verbose = 0
	x = [model.add_var(var_type=BINARY) for i in range(num_var)]
	model.objective = minimize(xsum(x[i] for i in range(n)))

	# orthogonality to rows of stab constraints
	for row in range(m):
		weight = [0]*num_var
		supp = np.nonzero(stab[row,:])[0]
		for q in supp:
			weight[q] = 1
		cnt = 1
		for q in range(num_anc_stab):
			weight[n + row*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
		model+= xsum(weight[i] * x[i] for i in range(num_var)) == 0

	# odd overlap with logicOp constraint
	supp = np.nonzero(logicOp)[0]
	weight = [0]*num_var
	for q in supp:
		weight[q] = 1
	cnt = 1
	for q in range(num_anc_logical):
			weight[n + m*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
	model+= xsum(weight[i] * x[i] for i in range(num_var)) == 1

	model.optimize()
	if model.status == OptimizationStatus.OPTIMAL:
		opt_val = sum([x[i].x for i in range(n)])
	else:
		opt_val = 0

	return int(opt_val)

#------------------------------------------------------------------------------------------------------------------

def simulate_circuitZ(C, n_, lin_order):
    syndrome_history = []
    syndrome_map = {}
    state = np.zeros(n_, dtype=int)
    # need this for debugging
    err_cnt = 0
    syn_cnt = 0
    for gate in C:
        if gate[0]=='CNOT':
            assert(len(gate)==3)
            control = lin_order[gate[1]]
            target = lin_order[gate[2]]
            state[control] = (state[target] + state[control]) % 2
            continue
        if gate[0]=='PrepX':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            state[q]=0
            continue
        if gate[0]=='MeasX':
            assert(len(gate)==2)
            assert(gate[1][0]=='Xcheck')
            q = lin_order[gate[1]]
            syndrome_history.append(state[q])
            if gate[1] in syndrome_map:
                syndrome_map[gate[1]].append(syn_cnt)
            else:
                syndrome_map[gate[1]] = [syn_cnt]
            syn_cnt+=1
            continue
        if gate[0] in ['Z','Y']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['ZX', 'YX']:
            err_cnt+=1
            assert(len(gate)==3)
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['XZ','XY']:
            err_cnt+=1
            assert(len(gate)==3)
            q = lin_order[gate[2]]
            state[q] = (state[q] + 1) % 2
            continue            
        if gate[0] in ['ZZ','YY','YZ','ZY']:
            err_cnt+=1
            assert(len(gate)==3)
            q1 = lin_order[gate[1]]
            q2 = lin_order[gate[2]]
            state[q1] = (state[q1] + 1) % 2
            state[q2] = (state[q2] + 1) % 2
            continue   
    return np.array(syndrome_history,dtype=int), state, syndrome_map, err_cnt

def simulate_circuitX(C, n_, lin_order):
    syndrome_history = []
    syndrome_map = {}
    state = np.zeros(n_, dtype=int)
    err_cnt = 0
    syn_cnt = 0
    for gate in C:
        if gate[0]=='CNOT':
            assert(len(gate)==3)
            control = lin_order[gate[1]]
            target = lin_order[gate[2]]
            state[target] = (state[target] + state[control]) % 2
            continue
        if gate[0]=='PrepZ':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            state[q]=0
            continue   
        if gate[0]=='MeasZ':
            assert(len(gate)==2)
            assert(gate[1][0]=='Zcheck')
            q = lin_order[gate[1]]
            syndrome_history.append(state[q])
            if gate[1] in syndrome_map:
                syndrome_map[gate[1]].append(syn_cnt)
            else:
                syndrome_map[gate[1]] = [syn_cnt]
            syn_cnt+=1
            continue
        if gate[0] in ['X','Y']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['XZ', 'YZ']:
            err_cnt+=1
            assert(len(gate)==3)
            q = lin_order[gate[1]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['ZX','ZY']:
            err_cnt+=1
            assert(len(gate)==3)
            q = lin_order[gate[2]]
            state[q] = (state[q] + 1) % 2
            continue

        if gate[0] in ['XX','YY','XY','YX']:
            err_cnt+=1
            assert(len(gate)==3)
            q1 = lin_order[gate[1]]
            q2 = lin_order[gate[2]]
            state[q1] = (state[q1] + 1) % 2
            state[q2] = (state[q2] + 1) % 2
            continue
    return np.array(syndrome_history,dtype=int),state,syndrome_map,err_cnt  


#-----------------------------------------------------------------------------------------------------------------

def generate_noisy_circuit(error_rate_init, error_rate_idle,error_rate_cnot,error_rate_meas, circuit):
    circ = []
    err_cnt = 0

    for gate in circuit:
        assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ'])
        if gate[0]=='MeasX':
            if np.random.uniform()<=error_rate_meas:
                circ.append(('Z',gate[1]))
                err_cnt+=1
            circ.append(gate)
            continue
        if gate[0]=='IDLE':
            if np.random.uniform()<=error_rate_idle:
                ptype = np.random.randint(3)
                if ptype==0:
                    circ.append(('X',gate[1]))
                if ptype==1:
                    circ.append(('Y',gate[1]))
                if ptype==2:
                    circ.append(('Z',gate[1]))
                err_cnt+=1
            continue
        if gate[0]=='PrepX':
            circ.append(gate)
            if np.random.uniform()<=error_rate_init:
                circ.append(('Z',gate[1]))
                err_cnt+=1
            continue
        if gate[0]=='CNOT':
            circ.append(gate)
            if np.random.uniform()<=error_rate_cnot:
                error_type = np.random.randint(15)
                if error_type==0:
                    circ.append(('X',gate[1]))
                    err_cnt+=1
                    continue
                if error_type==1:
                    circ.append(('Y',gate[1]))
                    err_cnt+=1
                    continue
                if error_type==2:
                    circ.append(('Z',gate[1]))
                    err_cnt+=1
                    continue
                if error_type==3:
                    circ.append(('X',gate[2]))
                    err_cnt+=1
                    continue
                if error_type==4:
                    circ.append(('Y',gate[2]))
                    err_cnt+=1
                    continue
                if error_type==5:
                    circ.append(('Z',gate[2]))
                    err_cnt+=1
                    continue
                if error_type==6:
                    circ.append(('XX',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==7:
                    circ.append(('YY',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==8:
                    circ.append(('ZZ',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==9:
                    circ.append(('XY',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==10:
                    circ.append(('YX',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==11:
                    circ.append(('YZ',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==12:
                    circ.append(('ZY',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==13:
                    circ.append(('XZ',gate[1],gate[2]))
                    err_cnt+=1
                    continue
                if error_type==14:
                    circ.append(('ZX',gate[1],gate[2]))
                    err_cnt+=1
                    continue
        if gate[0]=='PrepZ':
            circ.append(gate)
            if np.random.uniform()<=error_rate_init:
                circ.append(('X',gate[1]))
                err_cnt+=1
            continue
        if gate[0]=='MeasZ':
            if np.random.uniform()<=error_rate_meas:
                circ.append(('X',gate[1]))
                err_cnt+=1
            circ.append(gate)
            continue

    return circ




































