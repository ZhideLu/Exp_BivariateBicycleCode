# Portions of this file were adapted from [BivariateBicycleCodes] 
# (https://github.com/sbravyi/BivariateBicycleCodes/tree/main) licensed under the Apache License 2.0.

# Without reset


import numpy as np
from mip import Model, xsum, minimize, BINARY
from bposd.css import css_code
from mip import OptimizationStatus
from scipy.sparse import coo_matrix
from scipy.sparse import hstack 

#-------------------------------------------------------------------------------------------------------------------------
num_end = 1 ;
#-------------------------------------------------------------------------------------------------------------------------
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

# computes the minimum Hamming weight of a binary vector x such that 
# stab @ x = 0 mod 2
# logicOp @ x = 1 mod 2
# here stab is a binary matrix and logicOp is a binary vector
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
#-------------------------------------------------------------------------------------------------------------------------

# [[18, 4, 4]]
ell,m = 3, 3
a1,a2,a3 = 1, 0, 2
b1,b2,b3 = 1, 0, 2

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


##-------------------------------------------------------------------------------------------------------------------------
def get_SM_circuit_parallel( remove_X_list, remove_Z_list, lin_order, data_qubits, Xchecks, Zchecks, nbs, sX, sZ ):
    cycle = [] 
    # round 0: prep xchecks, CZ zchecks and data
    ##----------------------------------------------------------------------------------------------------------------------
    t = 0
    for q in Xchecks:
        cycle.append(('X_Prep',q))
    for q in Zchecks:
        cycle.append(('Z_Prep',q))
    assert(sZ[t]=='idle')
    assert(sX[t]=='idle')
    
    for q in Xchecks:
        cycle.append(('H',q))  
    for q in Zchecks:
        cycle.append(('H',q))  
    
    for q in data_qubits[:n2]:  # L
        cycle.append(('H',q))
    ##----------------------------------------------------------------------------------------------------------------------
    t = 1
    data_qubits_cnoted_in_this_round = []
    assert(not(sZ[t]=='idle'))
    for target in Zchecks:
        direction = sZ[t]
        control = nbs[(target,direction)]
        
        data_qubits_cnoted_in_this_round.append(control)
        cycle.append(('CZ',control,target))
        
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            cycle.append(('IDLE',q))
    ##----------------------------------------------------------------------------------------------------------------------
    t = 2
    data_qubits_cnoted_in_this_round = []
    
    assert(not(sX[t]=='idle'))
    for control in Xchecks:
        direction = sX[t]
        target = nbs[(control,direction)]
        
        data_qubits_cnoted_in_this_round.append(target)   
        cycle.append(('CZ',control,target))
        
    assert(not(sZ[t]=='idle'))
    for target in Zchecks:
        direction = sZ[t]
        control = nbs[(target,direction)] ;
        
        data_qubits_cnoted_in_this_round.append(control) 
        cycle.append(('CZ',control,target))
       
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            cycle.append(('IDLE',q))
    
    for q in data_qubits:  
        cycle.append(('H',q))
    ##----------------------------------------------------------------------------------------------------------------------
    # round 3-5: CZ xchecks and data, CZ zchecks and data
    for t in range(3,6):
        data_qubits_cnoted_in_this_round = []
        
        assert(not(sX[t]=='idle'))
        for control in Xchecks:
            direction = sX[t]
            target = nbs[(control,direction)]
            
            data_qubits_cnoted_in_this_round.append(target)   
            cycle.append(('CZ',control,target))
            
        assert(not(sZ[t]=='idle'))
        for target in Zchecks:
            direction = sZ[t]
            control = nbs[(target,direction)] ;
            
            data_qubits_cnoted_in_this_round.append(control) 
            cycle.append(('CZ',control,target))
           
        for q in data_qubits:
            if not(q in data_qubits_cnoted_in_this_round):
                cycle.append(('IDLE',q))
    
    for q in data_qubits:  
        cycle.append(('H',q))
    ##----------------------------------------------------------------------------------------------------------------------
    t = 6
    data_qubits_cnoted_in_this_round = []
    
    assert(not(sX[t]=='idle'))
    for control in Xchecks:
        direction = sX[t]
        target = nbs[(control,direction)]
        
        data_qubits_cnoted_in_this_round.append(target)   
        cycle.append(('CZ',control,target))
        
    assert(not(sZ[t]=='idle'))
    for target in Zchecks:
        direction = sZ[t]
        control = nbs[(target,direction)] ;
        
        data_qubits_cnoted_in_this_round.append(control) 
        cycle.append(('CZ',control,target))
       
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            cycle.append(('IDLE',q))
    ##----------------------------------------------------------------------------------------------------------------------
    # round 7: CZ xchecks and data, measure Z checks
    t=7
    assert(not(sX[t]=='idle'))
    data_qubits_cnoted_in_this_round = []
    for control in Xchecks:
        direction = sX[t]
        target = nbs[(control,direction)]
    
        cycle.append(('CZ',control,target))
        data_qubits_cnoted_in_this_round.append(target)
    for q in data_qubits:
        if not(q in data_qubits_cnoted_in_this_round):
            cycle.append(('IDLE',q))
    ##----------------------------------------------------------------------------------------------------------------------
    for q in Xchecks:
        cycle.append(('H',q))  
    for q in Zchecks:
        cycle.append(('H',q))  
    for q in data_qubits[:n2]:   # L
        cycle.append(('H',q))
    ##----------------------------------------------------------------------------------------------------------------------
    # round 8: all data qubits are idle, Prep Z checks, Meas X checks
    for q in Xchecks:
        cycle.append(('X_Meas',q))
    for q in Zchecks:
        cycle.append(('Z_Meas',q))
    for q in data_qubits:
        cycle.append(('DD',q))
    
    return cycle
#-------------------------------------------------------------------------------------------------------------------------
def get_SM_circuit_separate( remove_X_list, remove_Z_list, lin_order, data_qubits, Xchecks, Zchecks, nbs, sX, sZ ):
    cycle_new = [] ;
    # implement syndrome measurements using the circuit containing 12 layers of CZ gates
    
    for q in Xchecks:
        cycle_new.append(('X_Prep',q))
    for q in Zchecks:
        cycle_new.append(('Z_Prep',q))
    
    for q in Xchecks:
        cycle_new.append(('H',q))   
    
    for q in data_qubits:
        cycle_new.append(('H',q))
    
    # first measure all X checks
    for t in range(8):
        data_qubits_cnoted_in_this_round = []
        if not(sX[t]=='idle'):
            for control in Xchecks:
                direction = sX[t]
                target = nbs[(control,direction)]
                
                data_qubits_cnoted_in_this_round.append(target)   
                cycle_new.append(('CZ',control,target))
    
            for q in data_qubits:
                if not(q in data_qubits_cnoted_in_this_round):
                    cycle_new.append(('IDLE',q))
    
    for q in Xchecks:
        cycle_new.append(('H',q))   
    
    for q in data_qubits:
        cycle_new.append(('H',q))
        
    for q in Xchecks:
        cycle_new.append(('X_Meas',q))
    
    
    for q in Zchecks:
        cycle_new.append(('H',q))
        
    # next measure all Z checks
    for t in range(8):
        data_qubits_cnoted_in_this_round = []
        if not(sZ[t]=='idle'):
            for target in Zchecks:
                direction = sZ[t]
                control = nbs[(target,direction)]
                
                data_qubits_cnoted_in_this_round.append( control )   ###
                cycle_new.append(('CZ',control,target))
    
            for q in data_qubits:
                if not(q in data_qubits_cnoted_in_this_round):
                    cycle_new.append(('IDLE',q))
    
    for q in Zchecks:
        cycle_new.append(('H',q))
    
    for q in Zchecks:
        cycle_new.append(('Z_Meas',q))

    return cycle_new
#-------------------------------------------------------------------------------------------------------------------------
# Connections of edges in the Tanner graph
def get_connection_Tanner(remove_X_list, remove_Z_list, n2):
	# Give a name to each qubit
	# Define a linear order on the set of qubits
	lin_order = {}
	data_qubits = []
	Xchecks = []
	Zchecks = []    
	cnt = 0
	for i in np.delete(range(n2), remove_X_list):
		node_name = ('Xcheck', i)
		Xchecks.append(node_name)
		lin_order[node_name] = cnt
		cnt += 1
	
	for i in range(n2):
		node_name = ('data_left', i)
		data_qubits.append(node_name)
		lin_order[node_name] = cnt
		cnt += 1
	for i in range(n2):
		node_name = ('data_right', i)
		data_qubits.append(node_name)
		lin_order[node_name] = cnt
		cnt += 1
	
	for i in np.delete(range(n2), remove_Z_list):
		node_name = ('Zcheck', i)
		Zchecks.append(node_name)
		lin_order[node_name] = cnt
		cnt += 1    

	# compute the list of neighbors of each check qubit in the Tanner graph
	nbs = {}
	for i in np.delete(range(n2), remove_X_list):
		check_name = ('Xcheck',i)
		# left data qubits
		nbs[(check_name,0)] = ('data_left',np.nonzero(A1[i,:])[0][0])
		nbs[(check_name,1)] = ('data_left',np.nonzero(A2[i,:])[0][0])
		nbs[(check_name,2)] = ('data_left',np.nonzero(A3[i,:])[0][0])
		# right data qubits
		nbs[(check_name,3)] = ('data_right',np.nonzero(B1[i,:])[0][0])
		nbs[(check_name,4)] = ('data_right',np.nonzero(B2[i,:])[0][0])
		nbs[(check_name,5)] = ('data_right',np.nonzero(B3[i,:])[0][0])
	
	for i in np.delete(range(n2), remove_Z_list):
		check_name = ('Zcheck',i)
		# left data qubits
		nbs[(check_name,0)] = ('data_left',np.nonzero(B1[:,i])[0][0])
		nbs[(check_name,1)] = ('data_left',np.nonzero(B2[:,i])[0][0])
		nbs[(check_name,2)] = ('data_left',np.nonzero(B3[:,i])[0][0])
		# right data qubits
		nbs[(check_name,3)] = ('data_right',np.nonzero(A1[:,i])[0][0])
		nbs[(check_name,4)] = ('data_right',np.nonzero(A2[:,i])[0][0])
		nbs[(check_name,5)] = ('data_right',np.nonzero(A3[:,i])[0][0])
		
	return  lin_order, data_qubits, Xchecks, Zchecks, nbs

def get_layout(nbs, remove_X_list, remove_Z_list):
	assert(remove_X_list == [])
	assert(remove_Z_list == [])
	#以 L2 为起点的横向轨迹
	hor_traj = []
	hor_traj.append(('data_left', 2)) ;
	for i in range(4*ell):
		
		q = [key for key, value in nbs.items() if value == hor_traj[-1] if key[1] == 2 if key[0][0] == "Zcheck"]
		hor_traj.append( q[0][0] )
		
		if nbs[(hor_traj[-1],1)] == hor_traj[0]: 
			break    
		hor_traj.append( nbs[(hor_traj[-1],1)] ) ;
	#以 L2 为起点的纵向轨迹
	ver_traj = []
	ver_traj.append(('data_left', 2)) ;
	for i in range(4*ell):
		q = [key for key, value in nbs.items() if value == ver_traj[-1] if key[1] == 1 if key[0][0] == "Xcheck"]
		ver_traj.append( q[0][0] )
		
		if nbs[(ver_traj[-1], 2)] == ver_traj[0]: 
			break    
		ver_traj.append( nbs[(ver_traj[-1], 2)] ) ;
#---------------------------------------------------------------------------------------------------
	layout_list = [] ;
	for s in ver_traj:
		traj = [] ;
		traj.append( s ) ;
		if s[0] == 'data_left' :
			for i in range(4*ell):
				q = [key for key, value in nbs.items() if value == traj[-1] if key[1] == 2 
					 if key[0][0] == "Zcheck"] ;
				traj.append( q[0][0] )
	
				if nbs[(traj[-1], 1)] == traj[0]: 
					break
	
				traj.append( nbs[(traj[-1], 1)] ) ;
			layout_list.append(traj);  
#---------------------------------------------------------------------------------------------------
		if s[0] == 'Xcheck' :
			for i in range(4*ell):
				
				traj.append( nbs[(traj[-1], 5)] ) ;
				
				q = [key for key, value in nbs.items() if value == traj[-1] if key[1] == 4 
					 if key[0][0] == "Xcheck"] ;
				
				if q[0][0] == traj[0]: 
					break                
				traj.append( q[0][0] ) ;   
					  
			layout_list.append(traj);
	return layout_list
#---------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------

def get_Set_noisy_circuits_logical_X(cycle_repeated, error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip):
    # print('Generating noisy circuits with a singe faulty operation...')
    # for the logical X state
    X_Prob = []
    X_circuits = []
    head = []
    tail = cycle_repeated.copy()
    for gate in cycle_repeated:
        assert(gate[0] in ['X_Prep', 'Z_Prep', 'IDLE', 'H', 'CZ', 'X_Meas', 'Z_Meas', 'DD', 'final'])
        
        if gate[0]=='Z_Meas':
            assert(len(gate)==2)
            X_circuits.append(head + [('X',gate[1])] + [tail[0]] + [('X',gate[1])] + tail[1:])
            X_Prob.append(error_rate_meas)
    
        if gate[0]=='X_Meas':
            assert(len(gate)==2)
            X_circuits.append(head + [('X',gate[1])] + [tail[0]] + [('X',gate[1])] + tail[1:])
            X_Prob.append(error_rate_meas)
    
        # move the gate from tail to head
        head.append(gate)
        tail.pop(0)
        assert(cycle_repeated==(head+tail))
        # if gate[0]=='X_Prep':
        # 	assert(len(gate)==2)
        # 	X_circuits.append(head + [('X',gate[1])] + tail)
        # 	X_Prob.append(error_rate_init)  
        # if gate[0]=='Z_Prep':
        # 	assert(len(gate)==2)
        # 	X_circuits.append(head + [('X',gate[1])] + tail)
        # 	X_Prob.append(error_rate_init)    
        if gate[0]=='DD':
            assert(len(gate)==2)
            X_circuits.append(head + [('Z',gate[1])] + tail)
            X_Prob.append( error_DD_phaseflip )

        if gate[0]=='final':
            assert(len(gate)==2)
            X_circuits.append(head + [('Z',gate[1])] + tail)
            X_Prob.append( error_final )
    
        if gate[0]=='H':
            assert(len(gate)==2)
            X_circuits.append(head + [('Z',gate[1])] + tail)
            X_Prob.append(error_rate_H*1/3)
            X_circuits.append(head + [('X',gate[1])] + tail)
            X_Prob.append(error_rate_H*1/3)
            X_circuits.append(head + [('Y',gate[1])] + tail)
            X_Prob.append(error_rate_H*1/3)   
    
        if gate[0]=='IDLE':
            assert(len(gate)==2)
            X_circuits.append(head + [('Z',gate[1])] + tail)
            X_Prob.append(error_rate_idle*1/3)
            X_circuits.append(head + [('X',gate[1])] + tail)
            X_Prob.append(error_rate_idle*1/3)
            X_circuits.append(head + [('Y',gate[1])] + tail)
            X_Prob.append(error_rate_idle*1/3)        
        
        if gate[0]=='CZ':
            assert(len(gate)==3)
            # add error on the control qubit
            X_circuits.append(head + [('Z',gate[1])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('X',gate[1])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('Y',gate[1])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            
            # add error on the target qubit
            X_circuits.append(head + [('Z',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('X',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('Y',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            
            # add ZZ error on the control and the target qubits
            X_circuits.append(head + [('Z',gate[1])] + [('Z',gate[2])] + tail)     
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('Z',gate[1])] + [('X',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('Z',gate[1])] + [('Y',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('X',gate[1])] + [('Z',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('X',gate[1])] + [('X',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('X',gate[1])] + [('Y',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('Y',gate[1])] + [('Z',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('Y',gate[1])] + [('X',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
            X_circuits.append(head + [('Y',gate[1])] + [('Y',gate[2])] + tail)
            X_Prob.append(error_rate_cz*1/15)
    return X_circuits, X_Prob

def get_Set_noisy_circuits_logical_Z(cycle_repeated, error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip):
    # print('Generating noisy circuits with a singe faulty operation...')
    # for the logical Z state
    Z_Prob = []
    Z_circuits = []
    head = []
    tail = cycle_repeated.copy()
    for gate in cycle_repeated:
        assert(gate[0] in ['X_Prep', 'Z_Prep', 'IDLE', 'H', 'CZ', 'X_Meas', 'Z_Meas', 'DD','final'])
        
        if gate[0]=='Z_Meas':
            assert(len(gate)==2)
            Z_circuits.append(head + [('X',gate[1])] + [tail[0]] + [('X',gate[1])] + tail[1:])
            Z_Prob.append(error_rate_meas)
    
        if gate[0]=='X_Meas':
            assert(len(gate)==2)
            Z_circuits.append(head + [('X',gate[1])] + [tail[0]] + [('X',gate[1])] + tail[1:])
            Z_Prob.append(error_rate_meas)
    
        # move the gate from tail to head
        head.append(gate)
        tail.pop(0)
        assert(cycle_repeated==(head+tail))
        # if gate[0]=='X_Prep':
        # 	assert(len(gate)==2)
        # 	Z_circuits.append(head + [('X',gate[1])] + tail)
        # 	Z_Prob.append(error_rate_init)
        # if gate[0]=='Z_Prep':
        # 	assert(len(gate)==2)
        # 	Z_circuits.append(head + [('X',gate[1])] + tail)
        # 	Z_Prob.append(error_rate_init)
        if gate[0]=='DD':
            assert(len(gate)==2)
            Z_circuits.append(head + [('X',gate[1])] + tail)
            Z_Prob.append( error_DD_bitflip )
        
        if gate[0]=='final':
            assert(len(gate)==2)
            Z_circuits.append(head + [('X',gate[1])] + tail)
            Z_Prob.append(error_final)
    
        if gate[0]=='H':
            assert(len(gate)==2)
            Z_circuits.append(head + [('Z',gate[1])] + tail)
            Z_Prob.append(error_rate_H*1/3)
            Z_circuits.append(head + [('X',gate[1])] + tail)
            Z_Prob.append(error_rate_H*1/3)
            Z_circuits.append(head + [('Y',gate[1])] + tail)
            Z_Prob.append(error_rate_H*1/3)   
    
        if gate[0]=='IDLE':
            assert(len(gate)==2)
            Z_circuits.append(head + [('Z',gate[1])] + tail)
            Z_Prob.append(error_rate_idle*1/3)
            Z_circuits.append(head + [('X',gate[1])] + tail)
            Z_Prob.append(error_rate_idle*1/3)
            Z_circuits.append(head + [('Y',gate[1])] + tail)
            Z_Prob.append(error_rate_idle*1/3)        
        
        if gate[0]=='CZ':
            assert(len(gate)==3)
            # add error on the control qubit
            Z_circuits.append(head + [('Z',gate[1])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('X',gate[1])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('Y',gate[1])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            
            # add error on the target qubit
            Z_circuits.append(head + [('Z',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('X',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('Y',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            
            # add ZZ error on the control and the target qubits
            Z_circuits.append(head + [('Z',gate[1])] + [('Z',gate[2])] + tail)     
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('Z',gate[1])] + [('X',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('Z',gate[1])] + [('Y',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('X',gate[1])] + [('Z',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('X',gate[1])] + [('X',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('X',gate[1])] + [('Y',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('Y',gate[1])] + [('Z',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('Y',gate[1])] + [('X',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
            Z_circuits.append(head + [('Y',gate[1])] + [('Y',gate[2])] + tail)
            Z_Prob.append(error_rate_cz*1/15)
    return Z_circuits, Z_Prob

#------------------------------------------------------------------------------------------------------------
# we only look at the action of the circuit on a single error;  0 means no error, 1 means error
# for the logical X state
def simulate_circuit_Lx(C, lin_order, n, remove_X_list, remove_Z_list):
    syndrome_history = []
    syndrome_map = {}
    
    basis = ["Z"] * (n2-len(remove_X_list)) + ["X"] * 2*n2 + ["Z"] * (n2-len(remove_Z_list))  
    gate_accumulated_X = np.zeros(2*n-len(remove_X_list)-len(remove_Z_list), dtype=int) ;
    gate_accumulated_Z = np.zeros(2*n-len(remove_X_list)-len(remove_Z_list), dtype=int) ;
    # need this for debugging
    err_cnt = 0
    syn_cnt = 0

    for gate in C:
        if gate[0] == 'X_Prep':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            assert( basis[q] == "Z" )
            
            gate_accumulated_X[q] = 0 
            gate_accumulated_Z[q] = 0     
            continue            
        if gate[0] == 'Z_Prep':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            assert( basis[q] == "Z" )
            
            gate_accumulated_X[q] = 0 
            gate_accumulated_Z[q] = 0     
            continue
            
        if gate[0] == 'X_Meas':
            assert(len(gate)==2)
            assert(gate[1][0]=='Xcheck')
            q = lin_order[gate[1]]
            
            assert(basis[q] == "Z")
            syndrome_history.append( gate_accumulated_X[q] )
            
            if gate[1] in syndrome_map:
                syndrome_map[gate[1]].append(syn_cnt)
            else:
                syndrome_map[gate[1]] = [syn_cnt]
            syn_cnt+=1
            continue

        if gate[0]=='CZ':
            assert(len(gate)==3)
            control = lin_order[gate[1]]
            target = lin_order[gate[2]]
            
            gate_accumulated_Z[target] = (gate_accumulated_Z[target] + gate_accumulated_X[control]) %2
            gate_accumulated_Z[control] = (gate_accumulated_Z[control] + gate_accumulated_X[target]) %2          
            continue
        
        if gate[0]=='H':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
        
            basis[q] = list( set(["X", "Z"]) - set( basis[q]) )[0]
        
            gate_accumulated_Z[q], gate_accumulated_X[q] = gate_accumulated_X[q], gate_accumulated_Z[q]      
            continue

        if gate[0] in ['Z']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            gate_accumulated_Z[q] = (gate_accumulated_Z[q]+1) %2
            continue
        if gate[0] in ['Y']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            gate_accumulated_Z[q] = (gate_accumulated_Z[q]+1) %2
            gate_accumulated_X[q] = (gate_accumulated_X[q]+1) %2
            continue
        if gate[0] in ['X']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            gate_accumulated_X[q] = (gate_accumulated_X[q]+1) %2
            continue
    state = gate_accumulated_Z
    return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

# we only look at the action of the circuit on a single error;  0 means no error, 1 means error
# for the logical Z state
def simulate_circuit_Lz(C, lin_order, n, remove_X_list, remove_Z_list):
    syndrome_history = []
    syndrome_map = {}
    
    basis = ["Z"] * (n2-len(remove_X_list)) + ["Z"] * 2*n2 + ["Z"] * (n2-len(remove_Z_list))
    gate_accumulated_X = np.zeros(2*n-len(remove_X_list)-len(remove_Z_list), dtype=int) ;
    gate_accumulated_Z = np.zeros(2*n-len(remove_X_list)-len(remove_Z_list), dtype=int) ;

    # need this for debugging
    err_cnt = 0
    syn_cnt = 0

    for gate in C:
        if gate[0] == 'X_Prep':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            assert( basis[q] == "Z" )
            
            gate_accumulated_X[q] = 0 
            gate_accumulated_Z[q] = 0     
            continue            
        if gate[0] == 'Z_Prep':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            assert( basis[q] == "Z" )
            
            gate_accumulated_X[q] = 0 
            gate_accumulated_Z[q] = 0     
            continue
            
        if gate[0] == 'Z_Meas':
            assert(len(gate)==2)
            assert(gate[1][0]=='Zcheck')
            q = lin_order[gate[1]]
            
            assert(basis[q] == "Z")
            syndrome_history.append( gate_accumulated_X[q] )
            
            if gate[1] in syndrome_map:
                syndrome_map[gate[1]].append(syn_cnt)
            else:
                syndrome_map[gate[1]] = [syn_cnt]
            syn_cnt+=1
            continue

        if gate[0]=='CZ':
            assert(len(gate)==3)
            control = lin_order[gate[1]]
            target = lin_order[gate[2]]
            
            gate_accumulated_Z[target] = (gate_accumulated_Z[target] + gate_accumulated_X[control]) %2
            gate_accumulated_Z[control] = (gate_accumulated_Z[control] + gate_accumulated_X[target]) %2            
            continue
        
        if gate[0]=='H':
            assert(len(gate)==2)
            q = lin_order[gate[1]]
        
            basis[q] = list( set(["X", "Z"]) - set( basis[q]) )[0]
        
            gate_accumulated_Z[q], gate_accumulated_X[q] = gate_accumulated_X[q], gate_accumulated_Z[q]            
            continue

        if gate[0] in ['Z']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            gate_accumulated_Z[q] = (gate_accumulated_Z[q]+1) %2
            continue

        if gate[0] in ['Y']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            gate_accumulated_Z[q] = (gate_accumulated_Z[q]+1) %2
            gate_accumulated_X[q] = (gate_accumulated_X[q]+1) %2
            continue

        if gate[0] in ['X']:
            err_cnt+=1
            assert(len(gate)==2)
            q = lin_order[gate[1]]
            gate_accumulated_X[q] = (gate_accumulated_X[q]+1) %2
            continue
    state = gate_accumulated_X
    return np.array(syndrome_history, dtype=int), state, syndrome_map, err_cnt

#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def decoding_X_matrix(X_circuits, X_Prob, num_cycles, cycle_new, lin_order, n, k, data_qubits, Xchecks, lx, remove_X_list, remove_Z_list):  
    HXdict  = {}
    # print('Computing syndrome histories for single-fault circuits...')
    # for the logical X state
    cnt = 0
    for circ in X_circuits:
        syndrome_history,state,syndrome_map,err_cnt = simulate_circuit_Lx(circ + cycle_new*num_end, lin_order, n, remove_X_list, remove_Z_list)
        assert(err_cnt<=2)
        assert(len(syndrome_history) == ( int(n/2)-len(remove_X_list) ) * (num_cycles + num_end ))
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lx @ state_data_qubits) % 2
        
        # apply syndrome sparsification map
        syndrome_history_copy = syndrome_history.copy()
    
        if num_cycles == 1:
            for c in Xchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                for row in range(1, num_cycles + num_end):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
            syndrome_history%= 2
    
        if num_cycles > 1:
            for c in Xchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                syndrome_history[pos[-1]]+= syndrome_history[pos[-2]] + syndrome_history[pos[-3]]
                
                for row in range(2, num_cycles + num_end - 1):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-2]]
            syndrome_history%= 2    
    
        syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
        supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        if supp in HXdict:
            HXdict[supp].append(cnt)
        else:
            HXdict[supp]=[cnt]
        cnt+=1
    first_logical_rowX = ( int(n/2)-len(remove_X_list) ) * (num_cycles + num_end)
    # if a subset of columns of HX are equal, retain only one of these columns
    num_err = len(HXdict)
    print('Number of distinct syndrome histories for logical X =', num_err)
    HX = []
    HdecX = []
    channel_probsX = []
    for supp in HXdict:
        new_column = np.zeros(( ( int(n/2)-len(remove_X_list) ) * (num_cycles + num_end)+k,1),dtype=int)
        new_column_short = np.zeros(( ( int(n/2)-len(remove_X_list) ) * (num_cycles + num_end),1),dtype=int)
        new_column[list(supp),0] = 1
        new_column_short[:,0] = new_column[0:first_logical_rowX,0]
        HX.append(coo_matrix(new_column))
        HdecX.append(coo_matrix(new_column_short))
        channel_probsX.append(np.sum([X_Prob[i] for i in HXdict[supp]]))
    HX = hstack(HX)
    HdecX = hstack(HdecX)
    return channel_probsX, HX, HdecX, HXdict

def decoding_Z_matrix(Z_circuits, Z_Prob, num_cycles, cycle_new, lin_order, n, k, data_qubits, Zchecks, lz, remove_X_list, remove_Z_list):  
    HZdict  = {}
    # print('Computing syndrome histories for single-fault circuits...')
    # for the logical Z state
    cnt = 0
    for circ in Z_circuits:
        syndrome_history,state,syndrome_map,err_cnt = simulate_circuit_Lz(circ + cycle_new*num_end, lin_order, n, remove_X_list, remove_Z_list)
        assert(err_cnt<=2)
        assert(len(syndrome_history) == ( int(n/2)-len(remove_X_list) ) * (num_cycles + num_end ))
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lz @ state_data_qubits) % 2
        
        # apply syndrome sparsification map
        syndrome_history_copy = syndrome_history.copy()
    
        if num_cycles == 1:
            for c in Zchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                for row in range(1, num_cycles + num_end):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
            syndrome_history%= 2
    
        if num_cycles > 1:
            for c in Zchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                syndrome_history[pos[-1]]+= syndrome_history[pos[-2]] + syndrome_history[pos[-3]]
                
                for row in range(2, num_cycles + num_end - 1):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-2]]
            syndrome_history%= 2    
    
        syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
        supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        if supp in HZdict:
            HZdict[supp].append(cnt)
        else:
            HZdict[supp]=[cnt]
        cnt+=1
    first_logical_rowZ = ( int(n/2)-len(remove_Z_list) ) * (num_cycles + num_end)
    # if a subset of columns of HX are equal, retain only one of these columns
    num_err = len(HZdict)
    print('Number of distinct syndrome histories for logical Z =', num_err)
    HZ = []
    HdecZ = []
    channel_probsZ = []
    for supp in HZdict:
        new_column = np.zeros(( ( int(n/2)-len(remove_Z_list) ) * (num_cycles + num_end)+k,1),dtype=int)
        new_column_short = np.zeros(( ( int(n/2)-len(remove_Z_list) ) * (num_cycles + num_end),1),dtype=int)
        new_column[list(supp),0] = 1
        new_column_short[:,0] = new_column[0:first_logical_rowZ,0]
        HZ.append(coo_matrix(new_column))
        HdecZ.append(coo_matrix(new_column_short))
        channel_probsZ.append(np.sum([Z_Prob[i] for i in HZdict[supp]]))
    HZ = hstack(HZ)
    HdecZ = hstack(HdecZ)
    return channel_probsZ, HZ, HdecZ, HZdict
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def generate_noisy_circuit(error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip, circuit):
    circ = []
    err_cnt = 0
    
    for gate in circuit:
        assert(gate[0] in ['H','CZ','IDLE','X_Prep','Z_Prep','X_Meas','Z_Meas', 'DD', 'final'])
        if gate[0]=='X_Meas':
            temp = np.random.uniform() ;
            if temp <= error_rate_meas:
                circ.append(('X',gate[1]))
                circ.append(gate)
                circ.append(('X',gate[1]))
                err_cnt+=1
            if temp > error_rate_meas:
                circ.append(gate)
            continue
        if gate[0]=='Z_Meas':
            temp = np.random.uniform() ;
            if temp <= error_rate_meas:
                circ.append(('X',gate[1]))
                circ.append(gate)
                circ.append(('X',gate[1]))
                err_cnt+=1
            if temp > error_rate_meas:
                circ.append(gate)
            continue
        
        if gate[0]=='DD':
            temp = np.random.uniform() ;
            if temp <= error_DD_phaseflip:
                circ.append(('Z',gate[1]))   
            if temp <= error_DD_bitflip:
                circ.append(('X',gate[1]))
            continue
        
        if gate[0]=='final':
            temp = np.random.uniform() ;
            if temp <= error_final:
                circ.append(('X',gate[1]))
                circ.append(('Z',gate[1]))
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
        
        if gate[0]=='H':
            circ.append(gate)
            if np.random.uniform()<=error_rate_H:
                ptype = np.random.randint(3)
                if ptype==0:
                    circ.append(('X',gate[1]))
                if ptype==1:
                    circ.append(('Y',gate[1]))
                if ptype==2:
                    circ.append(('Z',gate[1]))
                err_cnt+=1
            continue
        # if gate[0]=='PrepX':
        # 	circ.append(gate)
        # 	if np.random.uniform()<=error_rate_init:
        # 		circ.append(('X',gate[1]))
        # 		err_cnt+=1
        # 	continue
        # if gate[0]=='PrepZ':
        # 	circ.append(gate)
        # 	if np.random.uniform()<=error_rate_init:
        # 		circ.append(('X',gate[1]))
        # 		err_cnt+=1
        # 	continue
        if gate[0]=='CZ':
            circ.append(gate)
            if np.random.uniform()<=error_rate_cz:
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
                    circ.append(('X',gate[1]))
                    circ.append(('X',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==7:
                    circ.append(('Y',gate[1]))
                    circ.append(('Y',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==8:
                    circ.append(('Z',gate[1]))
                    circ.append(('Z',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==9:
                    circ.append(('X',gate[1]))
                    circ.append(('Y',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==10:
                    circ.append(('Y',gate[1]))
                    circ.append(('X',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==11:
                    circ.append(('Y',gate[1]))
                    circ.append(('Z',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==12:
                    circ.append(('Z',gate[1]))
                    circ.append(('Y',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==13:
                    circ.append(('X',gate[1]))
                    circ.append(('Z',gate[2]))  
                    err_cnt+=1
                    continue
                if error_type==14:
                    circ.append(('Z',gate[1]))
                    circ.append(('X',gate[2]))  
                    err_cnt+=1
                    continue
    return circ

#------------------------------------------------------------------------------------------------------------
# Z-type error data
def simulate_syndrome_Lx(num_trials, num_cycles, cycle_new, cycle_append, error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip, lx, lin_order, data_qubits, Xchecks, remove_X_list, remove_Z_list):

    syndrome_history_X = [] ; final_logical_x_outcome = [] ;
    cycle_no_reset_repeated = (num_cycles-1) * cycle_new[2*n2-len(remove_X_list)-len(remove_Z_list):] + \
                cycle_new[2*n2-len(remove_X_list)-len(remove_Z_list) : -2*n2] + cycle_append  ;

    for trial in range(num_trials):
        circ = generate_noisy_circuit(error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, \
                                      error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip, cycle_no_reset_repeated)
        
        syndrome_history,state,syndrome_map,err_cnt = simulate_circuit_Lx(circ + cycle_new*num_end, lin_order, n, remove_X_list, remove_Z_list)
        
        assert(len(syndrome_history)== ( int(n/2)-len(remove_X_list) ) * (num_cycles + num_end))
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lx @ state_data_qubits) % 2  
        
        syndrome_history_copy = syndrome_history.copy()
        # measurement_history --->  detection events
        if num_cycles == 1:
            for c in Xchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                for row in range(1, num_cycles + num_end):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
            syndrome_history%= 2
        
        if num_cycles > 1:
            for c in Xchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                syndrome_history[pos[-1]]+= syndrome_history[pos[-2]] + syndrome_history[pos[-3]]
                
                for row in range(2, num_cycles + num_end - 1):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-2]]
            syndrome_history%= 2  
        
        syndrome_history_X.append(syndrome_history);  final_logical_x_outcome.append(syndrome_final_logical)

    return syndrome_history_X, final_logical_x_outcome
#---------------------------------------------------------------------------------------------------------------------
# X-type error data
def simulate_syndrome_Lz(num_trials, num_cycles, cycle_new, cycle_append, error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip, lz, lin_order, data_qubits, Zchecks, remove_X_list, remove_Z_list):
	
    syndrome_history_Z = [] ;  final_logical_z_outcome = [] ;
    cycle_no_reset_repeated = (num_cycles-1) * cycle_new[2*n2-len(remove_X_list)-len(remove_Z_list):] + \
                cycle_new[2*n2-len(remove_X_list)-len(remove_Z_list) : -2*n2] + cycle_append  ;
    
    for trial in range(num_trials):
        circ = generate_noisy_circuit(error_rate_init, error_rate_idle, error_rate_H, error_rate_cz, \
                                      error_rate_meas, error_final, error_DD_phaseflip, error_DD_bitflip, cycle_no_reset_repeated)
        
        syndrome_history,state,syndrome_map,err_cnt = simulate_circuit_Lz(circ + cycle_new*num_end, lin_order, n, remove_X_list, remove_Z_list)
        assert(len(syndrome_history)== (int(n/2)-len(remove_Z_list)) * (num_cycles + num_end))
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lz @ state_data_qubits) % 2  

        syndrome_history_copy = syndrome_history.copy()
        # measurement_history --->  detection events
        if num_cycles == 1:
            for c in Zchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                for row in range(1, num_cycles + num_end):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
            syndrome_history%= 2
    
        if num_cycles > 1:
            for c in Zchecks:
                pos = syndrome_map[c]
                assert(len(pos)==(num_cycles + num_end))
                syndrome_history[pos[-1]]+= syndrome_history[pos[-2]] + syndrome_history[pos[-3]]
                
                for row in range(2, num_cycles + num_end - 1):
                    syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-2]]
            syndrome_history%= 2  
    
        syndrome_history_Z.append(syndrome_history);   final_logical_z_outcome.append(syndrome_final_logical)
		
    return syndrome_history_Z, final_logical_z_outcome
#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------


## error decoding
def error_decoding(num_trials, k, bpd, mat_H, list_syndrome_history, final_logical_outcome ):
    # list_low_weight_error = [] ;
    good_trials = 0
    good_list = [0] * k ;
    
    no_correction_good_list = [0] * k ;
    no_correction_good_trials = 0
    
    for trial in range(num_trials):
        
        syndrome_history = list_syndrome_history[trial]
        syndrome_final_logical = final_logical_outcome[trial]
        
        no_correction_good_list += ( syndrome_final_logical == [0] * k   )
        if np.array_equal( syndrome_final_logical, [0] * k  ):
            assert(  all(cc == 0 for cc in syndrome_final_logical )  )
            no_correction_good_trials+=1
    
        bpd.decode(syndrome_history)
        low_weight_error = bpd.osdw_decoding
        # list_low_weight_error.append(low_weight_error)
        syndrome_history_augmented_guessed = (mat_H @ low_weight_error) % 2
        syndrome_final_logical_guessed = syndrome_history_augmented_guessed[len(syndrome_history):(len(syndrome_history)+k)]  
        
        ec_result = np.array_equal(syndrome_final_logical_guessed, syndrome_final_logical)
        correct = (syndrome_final_logical_guessed + syndrome_final_logical)   % 2
        
        good_list += ( correct == [0] * k   )
    
        if ec_result:
            assert(  all(cc == 0 for cc in correct )  )
            good_trials+=1

    return no_correction_good_trials, no_correction_good_list, good_trials, good_list













