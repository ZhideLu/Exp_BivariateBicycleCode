import numpy as np
import random

from itertools import combinations
from scipy.optimize import fsolve
from itertools import chain
from functools import reduce

from tabulate import tabulate
from scipy.sparse import coo_matrix, hstack 

from bposd.css import css_code
from ldpc import bposd_decoder, mod2

from mip import Model, xsum, minimize, BINARY, OptimizationStatus
import pickle
import itertools

from exp_ldpc import rank2, distance_test, simulate_circuitZ, simulate_circuitX, generate_noisy_circuit

#------------------------------------------------------------------------------------------------------------------------------
# Z-type error data
def simulate_Zerror_syndrome(num_trials, num_cycles, cycle_no_reset, error_rate_init, error_rate_idle, error_rate_cnot, error_rate_meas):
    
    Zerror_syndrome_history = [] ;
    Zerror_syndrome_final_logical = [] ;
    cycle_no_reset_repeated = num_cycles * cycle_no_reset    
    
    for trial in range(num_trials):
        circ = generate_noisy_circuit(error_rate_init, error_rate_idle, error_rate_cnot, error_rate_meas, cycle_no_reset_repeated)
        # correct Z errors 
        syndrome_history,state,syndrome_map,err_cntZ = simulate_circuitZ(circ + cycle_no_reset * num_end, n_, lin_order)  ###
    
        assert(len(syndrome_history)== (n2 - len(rx)) * (num_cycles + num_end))
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lx @ state_data_qubits) % 2  
        
        # recover the syndrome history in the case with reset
        for c in Xchecks:
            pos = syndrome_map[c]
            assert(len(pos)==(num_cycles + num_end))
            for row in range(num_cycles-2+num_end, -1, -1):
                if syndrome_history[pos[row]] == 1 :
                    syndrome_history[pos[row+1]] = 1 - syndrome_history[pos[row+1]]
    
        syndrome_history_copy = syndrome_history.copy()
        # measurement_history --->  detection events
        for c in Xchecks:
            pos = syndrome_map[c]
            assert(len(pos)==(num_cycles + num_end))
            for row in range(1,num_cycles + num_end):
                syndrome_history[pos[row]] = (syndrome_history[pos[row]] - syndrome_history_copy[pos[row-1]]) %2
    
        Zerror_syndrome_history.append(syndrome_history)
        Zerror_syndrome_final_logical.append(syndrome_final_logical)

    return Zerror_syndrome_history, Zerror_syndrome_final_logical

# X-type error data
def simulate_Xerror_syndrome(num_trials, num_cycles, cycle_no_reset, error_rate_init, error_rate_idle, error_rate_cnot, error_rate_meas):
    
    Xerror_syndrome_history = [] ;
    Xerror_syndrome_final_logical = [] ;
    cycle_no_reset_repeated = num_cycles * cycle_no_reset    
    
    for trial in range(num_trials):
        circ = generate_noisy_circuit(error_rate_init, error_rate_idle,error_rate_cnot,error_rate_meas, cycle_no_reset_repeated)
        # correct X errors 
        syndrome_history,state,syndrome_map,err_cntX = simulate_circuitX(circ + cycle_no_reset * num_end, n_, lin_order)  ###
        
        assert(len(syndrome_history)== (n2 - len(rx)) * (num_cycles + num_end))
        state_data_qubits = [state[lin_order[q]] for q in data_qubits]
        syndrome_final_logical = (lz @ state_data_qubits) % 2  
    
        # recover the syndrome history in the case with reset
        for c in Zchecks:
            pos = syndrome_map[c]
            assert(len(pos)==(num_cycles + num_end))
            for row in range(num_cycles-2+num_end, -1, -1):
                if syndrome_history[pos[row]] == 1 :
                    syndrome_history[pos[row+1]] = 1 - syndrome_history[pos[row+1]]
    
        syndrome_history_copy = syndrome_history.copy()
        # measurement_history --->  detection events
        for c in Zchecks:
            pos = syndrome_map[c]
            assert(len(pos)==(num_cycles + num_end))
            for row in range(1,num_cycles + num_end):
                syndrome_history[pos[row]] = (syndrome_history[pos[row]] - syndrome_history_copy[pos[row-1]]) %2
    
        Xerror_syndrome_history.append(syndrome_history)
        Xerror_syndrome_final_logical.append(syndrome_final_logical)
        
    return Xerror_syndrome_history, Xerror_syndrome_final_logical
#------------------------------------------------------------------------------------------------------------------------------

class revised_BB_code:
    def __init__(self, ell,m, a,b, rx,rz):
        self.ell = ell
        self.m = m
        self.a = a
        self.b = b
        self.rx = rx
        self.rz = rz
        self.A1,self.A2,self.A3,self.B1,self.B2,self.B3, self.hx, self.hz = self.generate_BB()
        self.k = 2*n2 - rank2(self.hx) - rank2(self.hz)
        self.lin_order, self.data_qubits, self.Xchecks, self.Zchecks = self.generate_lin_order()
        self.nbs = self.generate_nbs()
        self.lz, self.lx = self.parameter() 
    
    def generate_BB(self):
        I_ell = np.identity(self.ell, dtype = int)
        I_m = np.identity(self.m, dtype = int)
        I = np.identity(self.ell * self.m, dtype = int)
        x = {}
        y = {}
            
        for i in range(self.ell):
        	x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
        for i in range(self.m):
        	y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))
            
        A1 = x[self.a[0]%self.ell]
        A2 = y[self.a[1]%self.m] 
        A3 = y[self.a[2]%self.m]
        B1 = y[self.b[0]%self.m]
        B2 = x[self.b[1]%self.ell]
        B3 = x[self.b[2]%self.ell]
        
        A = (A1 + A2 + A3) % 2
        B = (B1 + B2 + B3) % 2

        AT = np.transpose(A);  BT = np.transpose(B)
        
        hx = np.hstack((A,B)); hz = np.hstack((BT,AT))
        hx = np.delete(hx, self.rx, axis=0); hz = np.delete(hz, self.rz, axis=0)

        return A1,A2,A3, B1,B2,B3, hx,hz

    def parameter(self):
        qcode=css_code(hx=self.hx, hz=self.hz)
        print('Testing CSS code...')
        qcode.test()
        print('Done')
        lz = qcode.lz
        lx = qcode.lx
        print('Computing code distance...')

        d = 2*n2
        for i in range(self.k):
        	w = distance_test(self.hz,lz[i,:])
        	d = min(d,w)
            
        for i in range(self.k):
        	w = distance_test(self.hx,lx[i,:])
        	d = min(d,w)
        print('Code parameters: n,k,d=',2*n2,self.k, d)
        return lz , lx
    
    def generate_lin_order(self):
        lin_order = {}; data_qubits = []
        Xchecks = [];   Zchecks = []
        
        cnt = 0
        for i in np.delete(range(n2), self.rx):
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
        
        for i in np.delete(range(n2), self.rz): 
            node_name = ('Zcheck', i)
            Zchecks.append(node_name)
            lin_order[node_name] = cnt
            cnt += 1
        return lin_order, data_qubits, Xchecks, Zchecks

    def generate_nbs(self):
        nbs = {}
        for i in np.delete(range(n2), self.rx):
        	check_name = ('Xcheck',i)
        	# left data qubits
        	nbs[(check_name,0)] = ('data_left',np.nonzero(self.A1[i,:])[0][0])
        	nbs[(check_name,1)] = ('data_left',np.nonzero(self.A2[i,:])[0][0])
        	nbs[(check_name,2)] = ('data_left',np.nonzero(self.A3[i,:])[0][0])
        	# right data qubits
        	nbs[(check_name,3)] = ('data_right',np.nonzero(self.B1[i,:])[0][0])
        	nbs[(check_name,4)] = ('data_right',np.nonzero(self.B2[i,:])[0][0])
        	nbs[(check_name,5)] = ('data_right',np.nonzero(self.B3[i,:])[0][0])
        
        for i in np.delete(range(n2), self.rz):
        	check_name = ('Zcheck',i)
        	# left data qubits
        	nbs[(check_name,0)] = ('data_left',np.nonzero(self.B1[:,i])[0][0])
        	nbs[(check_name,1)] = ('data_left',np.nonzero(self.B2[:,i])[0][0])
        	nbs[(check_name,2)] = ('data_left',np.nonzero(self.B3[:,i])[0][0])
        	# right data qubits
        	nbs[(check_name,3)] = ('data_right',np.nonzero(self.A1[:,i])[0][0])
        	nbs[(check_name,4)] = ('data_right',np.nonzero(self.A2[:,i])[0][0])
        	nbs[(check_name,5)] = ('data_right',np.nonzero(self.A3[:,i])[0][0])
        return nbs

#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
class SM_circuit:
    def __init__( self, nbs, lin_order, data_qubits, Xchecks, Zchecks, sX,sZ ):
        self.nbs = nbs
        self.lin_order = lin_order
        self.data_qubits = data_qubits
        self.Xchecks = Xchecks
        self.Zchecks = Zchecks
        self.sX = sX
        self.sZ = sZ
        self.U, self.cycle = self.cycle()

    def cycle(self):
        cycle = [] 
        U = np.identity(n_, dtype=int)
        # round 0: prep xchecks, CNOT zchecks and data
        t = 0
        for q in self.Xchecks:
        	cycle.append(('PrepX', q))
        for q in self.Zchecks:
        	cycle.append(('PrepZ',q))

        assert(self.sZ[t]=='idle')
        assert(self.sX[t]=='idle')
        for q in self.data_qubits:
        		cycle.append(('IDLE',q))
        
        t = 1        
        data_qubits_cnoted_in_this_round = []
        assert(not(self.sZ[t]=='idle'))
        for target in self.Zchecks:
        	direction = self.sZ[t]
        	control = self.nbs[(target,direction)]
        	U[self.lin_order[target],:] = (U[self.lin_order[target],:] + U[self.lin_order[control],:]) % 2
        	data_qubits_cnoted_in_this_round.append(control)
        	cycle.append(('CNOT', control, target))
        for q in self.data_qubits:
        	if not(q in data_qubits_cnoted_in_this_round):
        		cycle.append(('IDLE',q))
        
        # round 1-6: CNOT xchecks and data, CNOT zchecks and data
        for t in range(2,7):
            data_qubits_cnoted_in_this_round = []
            
            assert(not(self.sX[t] == 'idle'))
            for control in self.Xchecks:
                direction = self.sX[t]
                target = self.nbs[(control,direction)]
                U[self.lin_order[target],:] = (U[self.lin_order[target],:] + U[self.lin_order[control],:]) % 2
                data_qubits_cnoted_in_this_round.append(target)   ###
                cycle.append(('CNOT',control, target))
            assert(not(self.sZ[t]=='idle'))
            for target in self.Zchecks:
                direction = self.sZ[t]
                control = self.nbs[(target, direction)] ;
                U[self.lin_order[target],:] = (U[self.lin_order[target],:] + U[self.lin_order[control],:]) % 2
                data_qubits_cnoted_in_this_round.append(control)   ###
                cycle.append(('CNOT',control, target))
               
            for q in self.data_qubits:
                if not(q in data_qubits_cnoted_in_this_round):
            	    cycle.append(('IDLE',q))
        
        # round 7: CNOT xchecks and data, measure Z checks
        t = 7
        assert(not(self.sX[t]=='idle'))
        data_qubits_cnoted_in_this_round = []
        for control in self.Xchecks:
        	direction = self.sX[t]
        	target = self.nbs[(control,direction)]
        	U[self.lin_order[target],:] = (U[self.lin_order[target],:] + U[self.lin_order[control],:]) % 2
        	cycle.append(('CNOT',control,target))
        	data_qubits_cnoted_in_this_round.append(target)
        for q in self.data_qubits:
        	if not(q in data_qubits_cnoted_in_this_round):
        		cycle.append(('IDLE',q))
        
        # round 8: all data qubits are idle, Prep Z checks, Meas X checks
        for q in self.data_qubits:
        	cycle.append(('IDLE',q))
        for q in self.Xchecks:
        	cycle.append(('MeasX',q))
        for q in self.Zchecks:
        	cycle.append(('MeasZ',q))
        
        return U, cycle

    def test_sm_circuit(self):
        V = np.identity(n_, dtype=int)
        # first measure all X checks
        for t in range(8):
        	if not(self.sX[t]=='idle'):
        		for control in self.Xchecks:
        			direction = self.sX[t]
        			target = self.nbs[(control,direction)]
        			V[self.lin_order[target],:] = (V[self.lin_order[target],:] + V[self.lin_order[control],:]) % 2
        # next measure all Z checks
        for t in range(8):
        	if not(self.sZ[t]=='idle'):
        		for target in self.Zchecks:
        			direction = self.sZ[t]
        			control = self.nbs[(target,direction)]
        			V[self.lin_order[target],:] = (V[self.lin_order[target],:] + V[self.lin_order[control],:]) % 2
                    
        if np.array_equal(self.U, V):
        	print('circuit test: OK')
        else:
        	print('circuit test: FAIL')
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
ell,m = 3, 3
a = [1,0,2]; b = [1,0,2] ;
rx = [2,8]; rz = [3,4] ;
n2 = ell * m ;
n_ = 4*n2 - len(rx) - len(rz) ;
#----------------------------------------------------------------------------------
BB_code_18_4_4 = revised_BB_code(ell, m, a, b, rx, rz) ;
#----------------------------------------------------------------------------------
lz = BB_code_18_4_4.lz;  lx = BB_code_18_4_4.lx ;

# we should make each logical X anticommute with its corresponding logical Z.
lz[1] = (lz[1]+lz[2]) %2
lz[3] = (lz[3]+lz[0]) %2

lz_copy = lz.copy()
lz_copy[0] = lz[1] ;  lz_copy[1] = lz[0] ; lz_copy[2] = lz[3] ;  lz_copy[3] = lz[2] ;
lz = lz_copy
#----------------------------------------------------------------------------------
# sX = ['idle', 1, 4, 3, 5, 0, 2] ; sZ = [3, 5, 0, 1, 2, 4, 'idle'] ;
sX = ['idle', 'idle', 1, 4, 3, 5, 0, 2] ; sZ = ['idle', 3, 5, 0, 1, 2, 4, 'idle'] ;
k = BB_code_18_4_4.k;
lin_order, nbs = BB_code_18_4_4.lin_order, BB_code_18_4_4.nbs ;
data_qubits, Xchecks, Zchecks = BB_code_18_4_4.data_qubits, BB_code_18_4_4.Xchecks, BB_code_18_4_4.Zchecks ;
#--------------------------------------------------------------------------------------------------------------
SM_circuit_18_4_4 = SM_circuit( nbs, lin_order, data_qubits, Xchecks, Zchecks, sX,sZ ) ;
SM_circuit_18_4_4.test_sm_circuit()
cycle = SM_circuit_18_4_4.cycle ;
#------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
num_end = 1 ;

my_bp_method = "ms"
my_max_iter = 10000
my_osd_method = "osd_cs"
my_osd_order = 7
my_ms_scaling_factor = 0

class decoding_matrix_BB_code:
    def __init__(self, k, nbs,cycle,num_cycles, lin_order,data_qubits,Zchecks,Xchecks, error_rate_init,error_rate_idle,error_rate_cnot,error_rate_meas, lz, lx ):
        self.k = k
        self.nbs = nbs
        self.lin_order = lin_order
        self.data_qubits = data_qubits
        self.Zchecks = Zchecks
        self.Xchecks = Xchecks
        self.cycle = cycle
        self.num_cycles = num_cycles
        self.cycle_repeated = self.num_cycles * self.cycle

        self.error_rate_init = error_rate_init
        self.error_rate_idle = error_rate_idle
        self.error_rate_cnot = error_rate_cnot
        self.error_rate_meas = error_rate_meas

        self.lx = lx
        self.lz = lz
        
        self.circuitsZ, self.ProbZ = self.Z_noisy_circuit()
        self.circuitsX, self.ProbX = self.X_noisy_circuit()
        self.channel_probsX, self.HX, self.HdecX, self.HXdict = self.decoding_X_matrix()
        self.channel_probsZ, self.HZ, self.HdecZ, self.HZdict = self.decoding_Z_matrix()
        
        
        self.bpdX = bposd_decoder(
            self.HdecX,#the parity check matrix
            channel_probs=self.channel_probsX, #assign error_rate to each qubit. This will override "error_rate" input variable
            max_iter=my_max_iter, #the maximum number of iterations for BP)
            bp_method=my_bp_method,
            ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
            osd_method=my_osd_method, #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
            osd_order=my_osd_order #the osd search depth
            )
        self.bpdZ = bposd_decoder(
            self.HdecZ,
            channel_probs=self.channel_probsZ, 
            max_iter=my_max_iter, 
            bp_method=my_bp_method,
            ms_scaling_factor=my_ms_scaling_factor, 
            osd_method="osd_cs", 
            osd_order=my_osd_order 
            )

    def Z_noisy_circuit(self):
        ProbZ = []
        circuitsZ = []
        head = []
        tail = self.cycle_repeated.copy()
        for gate in self.cycle_repeated:
            assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ'])
            if gate[0]=='MeasX':
                assert(len(gate)==2)
                circuitsZ.append(head + [('Z',gate[1])] + tail)
                ProbZ.append(self.error_rate_meas)
            # move the gate from tail to head
            head.append(gate)
            tail.pop(0)
            assert( self.cycle_repeated == (head+tail) )
            
            if gate[0]=='PrepX':
                assert(len(gate)==2)
                circuitsZ.append(head + [('Z',gate[1])] + tail)
                ProbZ.append(self.error_rate_init)
            if gate[0]=='IDLE':
                assert(len(gate)==2)
                circuitsZ.append(head + [('Z',gate[1])] + tail)
                ProbZ.append(self.error_rate_idle*2/3)  
            if gate[0]=='CNOT':
                assert(len(gate)==3)
                # add error on the control qubit
                circuitsZ.append(head + [('Z',gate[1])] + tail)
                ProbZ.append(self.error_rate_cnot*4/15)
                # add error on the target qubit
                circuitsZ.append(head + [('Z',gate[2])] + tail)
                ProbZ.append(self.error_rate_cnot*4/15)
                # add ZZ error on the control and the target qubits
                circuitsZ.append(head + [('ZZ',gate[1],gate[2])] + tail)
                ProbZ.append(self.error_rate_cnot*4/15)
                
        num_errZ = len(circuitsZ)
        print('Number of Z noisy circuits=', num_errZ)
        return circuitsZ, ProbZ
            
    def X_noisy_circuit(self):
        ProbX = []
        circuitsX = []
        head = []
        tail = self.cycle_repeated.copy()
        for gate in self.cycle_repeated:
        	assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ'])
        	if gate[0]=='MeasZ':
        		assert(len(gate)==2)
        		circuitsX.append(head + [('X',gate[1])] + tail)
        		ProbX.append(self.error_rate_meas)
        	# move the gate from tail to head
        	head.append(gate)
        	tail.pop(0)
        	assert(self.cycle_repeated == (head+tail))
            
        	if gate[0]=='PrepZ':
        		assert(len(gate)==2)
        		circuitsX.append(head + [('X',gate[1])] + tail)
        		ProbX.append(self.error_rate_init)
        	if gate[0]=='IDLE':
        		assert(len(gate)==2)
        		circuitsX.append(head + [('X',gate[1])] + tail)
        		ProbX.append(self.error_rate_idle*2/3)
        	if gate[0]=='CNOT':
        		assert(len(gate)==3)
        		# add error on the control qubit
        		circuitsX.append(head + [('X',gate[1])] + tail)
        		ProbX.append(self.error_rate_cnot*4/15)
        		# add error on the target qubit
        		circuitsX.append(head + [('X',gate[2])] + tail)
        		ProbX.append(self.error_rate_cnot*4/15)
        		# add XX error on the control and the target qubits
        		circuitsX.append(head + [('XX',gate[1],gate[2])] + tail)
        		ProbX.append(self.error_rate_cnot*4/15)    
        num_errX=len(circuitsX)
        print('Number of X noisy circuits=',num_errX)
        return circuitsX, ProbX

    def decoding_X_matrix(self):
        HXdict  = {}        
        # execute each noisy circuit and compute the syndrome (we add one noiseless syndrome cycles at the end)
        cnt = 0
        for circ in self.circuitsX:
        	syndrome_history, state, syndrome_map, err_cnt = simulate_circuitX(circ + self.cycle * num_end, n_, self.lin_order)
        	assert(err_cnt==1)
        	assert(len(syndrome_history) == (n2 - len(rz)) * (self.num_cycles + num_end))
        	state_data_qubits = [state[self.lin_order[q]] for q in self.data_qubits]
        	syndrome_final_logical = (self.lz @ state_data_qubits) % 2
        	# apply syndrome sparsification map
        	syndrome_history_copy = syndrome_history.copy()
            
        	for c in self.Zchecks:
        		pos = syndrome_map[c]
        		assert(len(pos)==(self.num_cycles + num_end))
        		for row in range(1,self.num_cycles + num_end):
        			syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
        	syndrome_history%= 2
        	syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
        	supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        	if supp in HXdict:
        		HXdict[supp].append(cnt)
        	else:
        		HXdict[supp]=[cnt]
        	cnt+=1
        first_logical_rowX = (n2 - len(rz)) * (self.num_cycles + num_end) 
        num_errX = len(HXdict)
        print('Number of distinct X-syndrome histories=',num_errX)
        HX = [] ;   HdecX = [] ;   channel_probsX = [] ;
        for supp in HXdict:
        	new_column = np.zeros(((n2 - len(rz) ) * (self.num_cycles + num_end) + self.k, 1), dtype=int)
        	new_column_short = np.zeros(((n2 - len(rz) ) * (self.num_cycles + num_end), 1), dtype=int)
        	new_column[list(supp),0] = 1
        	new_column_short[:,0] = new_column[0:first_logical_rowX,0]
        	HX.append(coo_matrix(new_column))
        	HdecX.append(coo_matrix(new_column_short))
        	channel_probsX.append(np.sum([self.ProbX[i] for i in HXdict[supp]]))  ##
        HX = hstack(HX)
        HdecX = hstack(HdecX)
        # print('max col weight=',np.max(np.sum(HdecX,0)))
        # print('max row weight=',np.max(np.sum(HdecX,1)))
        return channel_probsX, HX, HdecX, HXdict

    def decoding_Z_matrix(self):
        HZdict  = {}
        cnt = 0
        for circ in self.circuitsZ:
        	syndrome_history,state,syndrome_map,err_cnt = simulate_circuitZ(circ + self.cycle * num_end, n_, self.lin_order)
        	assert(err_cnt==1)
        	assert(len(syndrome_history) == (n2 - len(rx) ) * (self.num_cycles + num_end ))
        	state_data_qubits = [state[self.lin_order[q]] for q in self.data_qubits]
        	syndrome_final_logical = (self.lx @ state_data_qubits) % 2
        	# apply syndrome sparsification map
        	syndrome_history_copy = syndrome_history.copy()
        	for c in self.Xchecks:
        		pos = syndrome_map[c]
        		assert(len(pos)==(self.num_cycles + num_end))
        		for row in range(1, self.num_cycles + num_end):
        			syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
        	syndrome_history%= 2
        	syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
        	supp = tuple(np.nonzero(syndrome_history_augmented)[0])
        	if supp in HZdict:
        		HZdict[supp].append(cnt)
        	else:
        		HZdict[supp]=[cnt]
        	cnt+=1
        
        first_logical_rowZ = (n2 - len(rx)) * (self.num_cycles + num_end)
        num_errZ = len(HZdict)
        print('Number of distinct Z-syndrome histories=',num_errZ)
        HZ = [] ; HdecZ = [] ;  channel_probsZ = []  ;
        for supp in HZdict:
        	new_column = np.zeros(( (n2 - len(rx)) * (self.num_cycles + num_end)+self.k, 1),dtype=int)
        	new_column_short = np.zeros(( (n2 - len(rx)) * (self.num_cycles + num_end), 1),dtype=int)
        	new_column[list(supp),0] = 1
        	new_column_short[:,0] = new_column[0:first_logical_rowZ,0]
        	HZ.append(coo_matrix(new_column))
        	HdecZ.append(coo_matrix(new_column_short))
        	channel_probsZ.append(np.sum([self.ProbZ[i] for i in HZdict[supp]]))
        HZ = hstack(HZ)
        HdecZ = hstack(HdecZ)
        # print('max col weight=',np.max(np.sum(HdecZ,0)))
        # print('max row weight=',np.max(np.sum(HdecZ,1)))  
        return channel_probsZ, HZ, HdecZ, HZdict

##————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# def equations(initial_guess, observe_values, cor_set, valid_cor_set, cover_set):
#     eqs = []
#     # 遍历每个x对应的方程
#     for observe, value in observe_values.items():
        
#         # 提取初始值
#         p_dict = {tuple(ele): initial_guess[j] for j, ele in enumerate(cor_set)}
        
#         equation = 0
        
#         # 遍历每个cover
#         cover = cover_set[observe] ;   
        
#         for comb in cover: 
#             equation += np.prod( [p_dict[tuple(s)] for s in comb] ) * np.prod( [1 - p_dict[tuple(s)] \
#                                                           for s in valid_cor_set[tuple(observe)] if not(s in comb) ] )

#         if len(observe) == 1:
#             for comb_3 in combinations(cover, 3):
#                 comb_3 = list( chain.from_iterable(comb_3) )
#                 equation += np.prod( [p_dict[tuple(s)] for s in comb_3] ) * np.prod( [1 - p_dict[tuple(s)] \
#                                                           for s in valid_cor_set[tuple(observe)] if not(s in comb_3) ] )
        
#         eqs.append(equation - value) 
        
#     return eqs
##------------------------------------------------------------------------------------------------------
# def equations_2(initial_guess, observe_values, cor_set, valid_cor_set, cover_set):
    
#     set_observe = [len(observe) for observe in observe_values] ;
#     set_value = list( observe_values.values() ) ;
#     eqs = []  
    
#     if len(cor_set) == 7:
#         assert(len( initial_guess )==7)
        
#         assert( set_observe == [3, 2, 2, 2, 1, 1, 1] )
    
#         assert(len( set_value )==7)
        
#         x123, x12, x23, x13, x1, x2, x3 = set_value[:7]
    
#         p123, p12, p23, p13, p1, p2, p3 = initial_guess[:7]
           
#         eqs = []         
#         eqs.append(p123 * (1-p1-p2-p3-p12-p23-p13) + p1 * p23 + p2 * p13 + p3 * p12 - x123) 
#         eqs.append( (p12+p123) * (1-(p1+p13+p2+p23)) + (p1+p13) * (p2+p23) - x12 ) 
#         eqs.append( (p23+p123) * (1-(p2+p12+p3+p13)) + (p2+p12) * (p3+p13) - x23 )
#         eqs.append( (p13+p123) * (1-(p1+p12+p3+p23)) + (p1+p12) * (p3+p23) - x13 )
#         eqs.append( p1 *(1-p12-p13-p123) + p12 *(1-p1-p13-p123) + p13 *(1-p1-p12-p123) + p123 *(1-p12-p13-p1) - x1 )
#         eqs.append( p2 *(1-p12-p23-p123) + p12 *(1-p2-p23-p123) + p23 *(1-p2-p12-p123) + p123 *(1-p12-p23-p2) - x2 )
#         eqs.append( p3 *(1-p13-p23-p123) + p13 *(1-p3-p23-p123) + p23 *(1-p3-p13-p123) + p123 *(1-p13-p23-p3) - x3 )

#     if len(cor_set) == 6:
#         assert(len( initial_guess )==6)
        
#         assert( set_observe == [3, 2, 2, 2, 1, 1] )
        
#         assert( len( set_value ) == 6 )

#         x123, x12, x23, x13, x1, x2 = set_value[:6]

#         p123, p12, p23, p1, p2, p3 = initial_guess[:6]
        
#         p13 = 0
              
#         eqs.append( p123 * (1-p1-p2-p3-p12-p23-p13) + p1 * p23 + p2 * p13 + p3 * p12 - x123 ) 
#         eqs.append( (p12+p123) * (1-(p1+p13+p2+p23)) + (p1+p13) * (p2+p23) - x12 ) 
#         eqs.append( (p23+p123) * (1-(p2+p12+p3+p13)) + (p2+p12) * (p3+p13) - x23 )
#         eqs.append( (p13+p123) * (1-(p1+p12+p3+p23)) + (p1+p12) * (p3+p23) - x13 )
#         eqs.append( p1 *(1-p12-p13-p123) + p12 *(1-p1-p13-p123) + p13 *(1-p1-p12-p123) + p123 *(1-p12-p13-p1) - x1 )
#         eqs.append( p2 *(1-p12-p23-p123) + p12 *(1-p2-p23-p123) + p23 *(1-p2-p12-p123) + p123 *(1-p12-p23-p2) - x2 )

#     if len(cor_set) == 5:
#         assert(len( initial_guess ) == 5)
        
#         assert( set_observe == [3, 2, 2, 2, 1] )
        
#         assert( len( set_value ) == 5 )

#         x123, x12, x23, x13, x1 = set_value[:5]

#         p123, p23, p1, p2, p3 = initial_guess[:5]
        
#         p13 = 0 ;  p12 = 0
              
#         eqs.append( p123 * (1-p1-p2-p3-p12-p23-p13) + p1 * p23 + p2 * p13 + p3 * p12 - x123 ) 
#         eqs.append( (p12+p123) * (1-(p1+p13+p2+p23)) + (p1+p13) * (p2+p23) - x12 ) 
#         eqs.append( (p23+p123) * (1-(p2+p12+p3+p13)) + (p2+p12) * (p3+p13) - x23 )
#         eqs.append( (p13+p123) * (1-(p1+p12+p3+p23)) + (p1+p12) * (p3+p23) - x13 )
#         eqs.append( p1 *(1-p12-p13-p123) + p12 *(1-p1-p13-p123) + p13 *(1-p1-p12-p123) + p123 *(1-p12-p13-p1) - x1 )

#     return eqs
##------------------------------------------------------------------------------------------------------
def equations_3( target, detect_expectation):

    ele = target ;
    
    x_1 = detect_expectation[(ele[0],)] ;
    x_2 = detect_expectation[(ele[1],)] ;
    x_12 = detect_expectation[(ele[0],ele[1])] ;
    
    q12 = p_ij(x_1, x_2, x_12) ;
    q1 = ((x_1) - q12)/(1 - 2*q12) ;
    q2 = ((x_2) - q12)/(1 - 2*q12) ;

    x_1 = detect_expectation[(ele[0],)] ;
    x_3 = detect_expectation[(ele[2],)] ;
    x_13 = detect_expectation[(ele[0],ele[2])] ;
    r13 = p_ij(x_1, x_3, x_13) ;
    r1 = ((x_1) - r13)/(1 - 2*r13) ;
    r3 = ((x_3) - r13)/(1 - 2*r13) ;

    x_2 = detect_expectation[(ele[1],)] ;
    x_3 = detect_expectation[(ele[2],)] ;
    x_23 = detect_expectation[(ele[1],ele[2])] ;
    t23 = p_ij(x_2, x_3, x_23) ;
    t2 = ((x_2) - t23)/(1 - 2*t23) ;
#--------------------------------------------------------------------- 
    a123 = detect_expectation[(ele[0],ele[1],ele[2])] ;
    
    p123 = 1/6 * ( 1 + 6*q12 + 2*q2 - 2*r1 - 2*r3 - 4*t2 -    \
                  np.sqrt(    (1 + 6*q12 + 2*q2 - 2*r1 - 2*r3 - 4*t2)**2      \
              + 12*(-a123-q1*q12-3*q12**2-2*q12*q2 + 2*q12*r1  + q2*r1+q12*r3+q1*t2+3*q12*t2-2*r1*t2   )                                                
                                         )                               \
                 ) ;
    return [p123]
#---------------------------------------------------------------------------------------------
def equations_4( target, detect_expectation):
    
    solutions = [] 
    deviation_list = []
    ele = target ;
    
    x_1 = detect_expectation[(ele[0],)] ;
    x_2 = detect_expectation[(ele[1],)] ;
    x_12 = detect_expectation[(ele[0],ele[1])] ;
    
    q12 = p_ij(x_1, x_2, x_12) ;
    q1 = ((x_1) - q12)/(1 - 2*q12) ;
    q2 = ((x_2) - q12)/(1 - 2*q12) ;

    x_1 = detect_expectation[(ele[0],)] ;
    x_3 = detect_expectation[(ele[2],)] ;
    x_13 = detect_expectation[(ele[0],ele[2])] ;
    r13 = p_ij(x_1, x_3, x_13) ;
    r1 = ((x_1) - r13)/(1 - 2*r13) ;
    r3 = ((x_3) - r13)/(1 - 2*r13) ;

    x_2 = detect_expectation[(ele[1],)] ;
    x_3 = detect_expectation[(ele[2],)] ;
    x_23 = detect_expectation[(ele[1],ele[2])] ;
    t23 = p_ij(x_2, x_3, x_23) ;
    t2 = ((x_2) - t23)/(1 - 2*t23) ;

    x_123 = detect_expectation[(ele[0],ele[1],ele[2])] ;

    for p123 in np.linspace(0.0001, 0.1, 10000):
        
        p12 = new_sub( q12, p123 )
        p13 = new_sub( new_sum(q1, q12), new_sum(r1, p123)   )
        p23 = new_sub( new_sum(q2, q12), new_sum(t2, p123)   )
        p1 = new_sub( new_sum(r1, p123),  q12  )
        p2 = new_sub( new_sum(t2, p123),  q12  )
        p3 = new_sub( r3, new_sub( new_sum(q2, q12), new_sum(t2, p123)   )   )

        temp4 = p123 * (1-p1) * (1-p2) * (1-p3) * (1-p12) * (1-p23) * (1-p13) \
           + p1 * p23 * (1-p123) * (1-p2) * (1-p3) * (1-p13) * (1-p12)   \
           + (1-p1) * p2 * p13 * (1-p123) * (1-p3) * (1-p23) * (1-p12)    \
           + (1-p1) * (1-p2) * p3 * p12 * (1-p123) * (1-p23) * (1-p13)   \
           + p123 * p1 * p23 * p2 * p13 * (1-p3) * (1-p12) \
           + p123 * p1 * p23 * p3 * p12 * (1-p2) * (1-p13) \
           + p123 * p2 * p13 * p3 * p12 * (1-p1) * (1-p23) \
           + (1-p123) * p1 * p23 * p2 * p13 * p3 * p12 - x_123 ;
        
        if ( -1e-5 < temp4 < 1e-5 )  :
            # print(temp4)
            deviation_list.append(temp4)
            solutions.append(p123)

    if len(solutions) == 0:
        print("Please try again, no solution.")
        print(target)
    
    sorted_indices = np.argsort(np.abs(deviation_list)) 
    solutions = [solutions[i] for i in sorted_indices ]

    return [solutions[0]]

#---------------------------------------------------------------------------------------------
def new_sum(p,q):
    return p + q - 2 * p * q
    
def new_sub(p,q):
    return (p - q)/(1 - 2*q)
    
def p_ij(x_i, x_j, x_ij):
    return (1 / 2) - (1 / 2) * np.sqrt(  1 - ( 4 * ( x_ij - x_i * x_j)) / (1 - 2 * x_i - 2 * x_j + 4 * x_ij )  )

##------------------------------------------------------------------------------------------------------
class correlation_nodes:
    def __init__(self, target, detection_list, detect_expectation):
        self.detection_list = detection_list
        self.detect_expectation = detect_expectation
        self.target = target
        self.cor_set = self.correlation_set()
        self.observe_set = self.observe_set()
        self.valid_cor_set = self.valid_correlation_set()
        self.cover_set = self.cover_set()
        self.observe_values = self.observe_values()
        
        self.initial_guess = [0.01] * len(self.cor_set)
        # self.solution = fsolve(equations, self.initial_guess, args = (self.observe_values, self.cor_set, self.valid_cor_set, self.cover_set), maxfev = 20000, xtol = 1e-6)

        self.solution = equations_4( self.target, self.detect_expectation)
    
    # correlation elements that overlap target 
    def correlation_set(self):
        temp1 = [(s,) for s in self.target] ;
        temp2 = [item for item in self.detection_list if len(set(item) & set(self.target)) == 2] ;
        
        if len(self.target) == 4:
            temp3 = [item for item in self.detection_list if len(set(item) & set(self.target)) == 3] ;
            temp = temp1 + temp2 + temp3
    
        if len(self.target) == 3:
            temp = temp1 + temp2      
        
        cor_set = [tuple( set( s ) & set(self.target) ) for s in [self.target] + temp  ] ; 
        cor_set = [list(s) for s in set(cor_set)]
        cor_set = sorted(cor_set, key=lambda x: (-len(x), x))  
        return cor_set
        
    # selected average observation of detection events  
    def observe_set(self):
        len_ = len(self.cor_set)
        observe_set = [list( s ) for r in range(len(self.target), 0, -1 ) for s in combinations(self.target, r)]
        observe_set = observe_set[:len_]
        return observe_set
    
    # for each average observation, valid_correlation_set that overlaps average observation
    def valid_correlation_set(self):
        valid_cor_set = {}
        for s in self.observe_set:
            valid_cor_set[ tuple(s) ] = [q for q in self.cor_set if any(x in q for x in set(s) )]
        return valid_cor_set

    # for each average observation, choose elements from valid_cor_set that can cover this observation once.
    def cover_set(self): 
        cover_set = {}
        
        for observe, valid in self.valid_cor_set.items():
            collection = []
            
            for r in range(1, len(observe) + 1): 
                for comb in itertools.combinations(valid, r):
                    
                    combined_set = set().union(*comb)  # 合并当前组合中的所有子集
            
                    if set(observe).issubset(combined_set):
                        # 确保没有元素被重复覆盖，即每个数字只出现一次
                        merged_comb = [item for s in comb for item in s ] ;
                        
                        if all( [merged_comb.count(i) == 1 for i in observe] ):
                            collection.append(comb)
                            
            cover_set[tuple( observe )] = collection
        return cover_set
    
    def observe_values(self):
        observe_values = {}
        for s in self.observe_set:
            observe_values[tuple(s)] = self.detect_expectation[tuple(s)]  # this is obbtained from experiment                
        return observe_values
##------------------------------------------------------------------------------------------------------
class DetectionToSyndromeProbability:
    def __init__(self, num_trials, num_cycles, syndrome_list, syndrome_history):
        self.num_trials = num_trials
        self.num_cycles = num_cycles
        self.syndrome_list = syndrome_list
        self.syndrome_history = syndrome_history
        self.syndrome_four_ele_property = self.get_syndrome_four_ele_property()
        self.detect_expectation = self.get_detect_expectation()
        self.syndrome_probability_list = self.get_syndrome_probability_list()
    
    def get_syndrome_four_ele_property(self):
        syndrome_four_ele_property = {} ;
        for key in self.syndrome_list:
            if len(key) == 4:
                syndrome_four_ele_property[key] = [None, None] ;
 
        target_4 = [item for item in self.syndrome_list if len(item) == 4] ;
        
        for ele in target_4:
            
            temp2 = [item for item in self.syndrome_list if len(set(item) & set(ele)) == 2] ;
            temp3 = [item for item in self.syndrome_list if len(set(item) & set(ele)) == 3] ;
        
            comb_valid_list_Two = [];
            for comb in combinations(ele, 2):
                
                if not any( set(comb).issubset(set(item)) for item in temp2 + temp3)  :
                    comb_valid_list_Two.append(comb)
            
            comb_valid_list_Three = [];
            for comb in combinations(ele, 3):
                if not any( set(comb).issubset(set(item)) for item in temp3)  :
                    comb_valid_list_Three.append(comb)
        
            comb_valid_list_TwoThree = [];
            for comb in combinations(ele, 2):
                if not any( set(comb).issubset(set(item)) for item in temp2 )  :
                    comb_valid_list_TwoThree.append(comb)
            
            if len(comb_valid_list_TwoThree) != 0:
                comb_valid_list_TwoThree = [ comb_valid_list_TwoThree[0] ] ;
                [ comb_valid_list_TwoThree.append( item ) for item in combinations(ele, 3) if set(comb_valid_list_TwoThree[0]).issubset(set(item))  ]
        
            if len(comb_valid_list_Two) != 0:
                syndrome_four_ele_property[ele][0] = "Two" ;
                syndrome_four_ele_property[ele][1] = comb_valid_list_Two ;
        
            if (len(comb_valid_list_Two) == 0) & (len(comb_valid_list_Three) != 0):
                syndrome_four_ele_property[ele][0] = "Three" ;
                syndrome_four_ele_property[ele][1] = comb_valid_list_Three ;
        
            if (len(comb_valid_list_Two) == 0) & (len(comb_valid_list_Three) == 0):       
                syndrome_four_ele_property[ele][0] = "TwoThree" ;
                syndrome_four_ele_property[ele][1] = comb_valid_list_TwoThree ;
    
        return syndrome_four_ele_property

    def get_detect_expectation(self):
        
        detect_expectation = {} ;
    
        for item in self.syndrome_list:
            temp_set = [tuple(comb) for r in range(2, len(item)+1)  for comb in combinations(item, r) ] ;
            for key in temp_set:    
                detect_expectation[key] = None ;
        
        for key in range( 7*(self.num_cycles + num_end) ):
            detect_expectation[(key,)] = None ;   ##
        
        for s in detect_expectation:
            detect_expectation[tuple(s)] =  \
                np.sum([ np.prod([ self.syndrome_history[trial][k] for k in s ]) for trial in range(self.num_trials) ]) / self.num_trials
        
        return detect_expectation
    
    ##注释
    def get_syndrome_probability_list(self):
    
        syndrome_probability_list = {} ;
        for key in self.syndrome_list:
            syndrome_probability_list[key] = None ;

        syndrome_probability_list[()] = 0 ; #
        target_4 = [item for item in self.syndrome_list if len(item) == 4] ;
        for ele in target_4:
            if self.syndrome_four_ele_property[ele][0] == "Two" :
                x_i = self.detect_expectation[(self.syndrome_four_ele_property[ele][1][0][0],)] ;
                x_j = self.detect_expectation[(self.syndrome_four_ele_property[ele][1][0][1],)] ;
                x_ij = self.detect_expectation[self.syndrome_four_ele_property[ele][1][0]] ;
                result = p_ij(x_i, x_j, x_ij)
                # print(result)
                syndrome_probability_list[ele] = result
        
            if self.syndrome_four_ele_property[ele][0] == "Three" :
                corr_three = correlation_nodes(self.syndrome_four_ele_property[ele][1][0], self.syndrome_list, self.detect_expectation) ;
                # print(corr_three.solution[0])
                syndrome_probability_list[ele] = corr_three.solution[0]
        
            if self.syndrome_four_ele_property[ele][0] == "TwoThree" :        
                corr_three_1 = correlation_nodes(self.syndrome_four_ele_property[ele][1][1], self.syndrome_list, self.detect_expectation) ;
                corr_three_2 = correlation_nodes(self.syndrome_four_ele_property[ele][1][2], self.syndrome_list, self.detect_expectation) ;
                
                x_i = self.detect_expectation[(self.syndrome_four_ele_property[ele][1][0][0],)] ;
                x_j = self.detect_expectation[(self.syndrome_four_ele_property[ele][1][0][1],)] ;
                x_ij = self.detect_expectation[self.syndrome_four_ele_property[ele][1][0]] ;
                result_0 = p_ij(x_i, x_j, x_ij) ;
                # result = corr_three_1.solution[0] + corr_three_2.solution[0] - result_0 ;
                result = new_sub( new_sum(corr_three_1.solution[0], corr_three_2.solution[0]), result_0 )
                
                syndrome_probability_list[ele] = result        
        #------------------------------------------------------------------------------------------------------------------------
        target_3 = [item for item in self.syndrome_list if len(item) == 3] ;
        for ele in target_3:
            corr_three = correlation_nodes(ele, self.syndrome_list, self.detect_expectation) ;
            temp_list = [syndrome_probability_list[tuple(item)] for item in target_4 if len(set(item) & set(ele)) == 3]
            if len(temp_list) == 0:
                syndrome_probability_list[tuple(ele)] = corr_three.solution[0]
            if len(temp_list) != 0:
                syndrome_probability_list[tuple(ele)] = new_sub( corr_three.solution[0], reduce(new_sum, temp_list) )
            
            # syndrome_probability_list[tuple(ele)] = corr_three.solution[0] - np.sum( temp_list )
        
        #------------------------------------------------------------------------------------------------------------------------
        target_2 = [item for item in self.syndrome_list if len(item) == 2] ;
        for ele in target_2:
            x_i = self.detect_expectation[(ele[0],)] ;
            x_j = self.detect_expectation[(ele[1],)] ;
            x_ij = self.detect_expectation[ele] ;
            result = p_ij(x_i, x_j, x_ij) ;
            temp_list_4 = [ syndrome_probability_list[tuple(item)] for item in target_4 if len(set(item) & set(ele)) == 2 ] ;
            temp_list_3 = [ syndrome_probability_list[tuple(item)] for item in target_3 if len(set(item) & set(ele)) == 2 ] ;
             # syndrome_probability_list[tuple(ele)] = result - np.sum( temp_list_4 ) - np.sum( temp_list_3 )
            if len(temp_list_4 + temp_list_3) == 0:
                syndrome_probability_list[tuple(ele)] = result
            if len(temp_list_4 + temp_list_3) != 0:
                syndrome_probability_list[tuple(ele)] = new_sub( result, reduce(new_sum, temp_list_4 + temp_list_3) )          
        #------------------------------------------------------------------------------------------------------------------------
        
        target_1 = [item for item in self.syndrome_list if len(item) == 1] ;
        
        for ele in target_1:
            temp =  [ syndrome_probability_list[tuple(item)] for item in target_4 if len(set(item) & set(ele)) == 1 ] +  \
                    [ syndrome_probability_list[tuple(item)] for item in target_3 if len(set(item) & set(ele)) == 1 ] +  \
                    [ syndrome_probability_list[tuple(item)] for item in target_2 if len(set(item) & set(ele)) == 1 ]
            
            P_sgima = reduce(new_sum, temp)
            # syndrome_probability_list[tuple(ele)] = ( self.detect_expectation[ele] - P_sgima ) / (1 - 2 * P_sgima ) ;
            syndrome_probability_list[tuple(ele)] = new_sub( self.detect_expectation[ele], P_sgima )
        
        return syndrome_probability_list
