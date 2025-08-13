import numpy as np
from itertools import product
import pickle
from packages import *

sys.path.append(os.path.abspath("../../"))
from functions_BB_code import distance_test, rank2, get_connection_Tanner, get_layout
num_level2 = 3 ;

# def f_locate(q, p): return [p.index('q_' + x) for x in q]
def f_locate(a, b):
    return [b.index(x) for x in a]
    
def average_GPC(num_qubits: int, GPC_list):
    channel = LeakyPauliChannel(is_single_qubit_channel=num_qubits == 1)
    for i in [0,1,16,17]:
        for j in [0,1,16,17]:
            for k in range(4**num_qubits):
                temp = np.mean([qq.get_prob_from_to(i, j, k) for qq in GPC_list])
                channel.add_transition(i, j, k, temp)
    return channel

def merge_GPC(num_qubits: int, GPC_list):
    len_ = len(GPC_list) ;
    channel = LeakyPauliChannel(is_single_qubit_channel=num_qubits == 1)
    for i in [0,1,16,17]:
        temp2 = np.sum([qq.get_prob_from_to(i, i, 0) for qq in GPC_list])
        # print(temp2-len_+1)
        channel.add_transition(i, i, 0, temp2-len_+1)
        
        for m in range(1, 4**num_qubits):
            temp = np.sum([qq.get_prob_from_to(i, i, m) for qq in GPC_list])
            channel.add_transition(i, i, m, temp)
    
        for j in [x for x in [0, 1, 16, 17] if x != i]:
            for k in range(4**num_qubits):
                temp = np.sum([qq.get_prob_from_to(i, j, k) for qq in GPC_list])
                channel.add_transition(i, j, k, temp)
    return channel

import re
mapping = {'R': 'data_right', 'L': 'data_left', 'X': 'Xcheck', 'Z': 'Zcheck'}

def convert_key(old_key):
    new_key = []
    for item in old_key:
        match = re.match(r'q_([A-Z])(\d+)', item)
        if match:
            letter, num = match.groups()
            new_key.append((mapping[letter], int(num)))
        else:
            new_key.append(item)
    return tuple(new_key)

def convert_key2(old_key):
    new_key = []
    for item in old_key:
        match = re.match(r'([A-Z])(\d+)', item)
        if match:
            letter, num = match.groups()
            new_key.append((mapping[letter], int(num)))
        else:
            new_key.append(item)
    return tuple(new_key)
    
def convert_item(item):

    match = re.match(r'c_([A-Z]\d+)_([A-Z]\d+)', item)
    if not match:
        return item 
    
    part1, part2 = match.groups()
    
    letter1, num1 = part1[0], int(part1[1:])
    part1_new = (mapping[letter1], num1)
    
    letter2, num2 = part2[0], int(part2[1:])
    part2_new = (mapping[letter2], num2)
    
    return (part1_new, part2_new)

def find_key_by_value(d, value, type_):
    if type_ == "CZ" :
        for k, v in d.items():
            if value in v:
                return k
            if (value[1], value[0]) in v:
                return k
 
    if type_ == "SQ" :
        for k, v in d.items():
            if value == v:
                return k
            if (value[1], value[0]) == v:
                return k
    return None          

# CZ layers
# [[18, 4, 4]]
ell,m = 3, 3 ; a1,a2,a3 = 1, 0, 2 ; b1,b2,b3 = 1, 0, 2
n = 2*m*ell;  n2 = m*ell ; remove_X_list = [2,8] ;  remove_Z_list = [3,4] ;
sX = ['idle', 'idle', 1, 4, 3, 5, 0, 2] ; sZ = ['idle', 3, 5, 0, 1, 2, 4, 'idle'] ;
lin_order, data_qubits, Xchecks, Zchecks, nbs = get_connection_Tanner(remove_X_list, remove_Z_list, n2)

CZ_layers = {} ;
t = 1
for target in Zchecks:
    direction = sZ[t] ; control = nbs[(target,direction)]
    CZ_layers.setdefault(t, []).append((control,target))
##----------------------------------------------------------------------------------------------------------------------
t = 2
for control in Xchecks:
    direction = sX[t] ; target = nbs[(control,direction)]
    CZ_layers.setdefault(t, []).append((control,target))
for target in Zchecks:
    direction = sZ[t] ; control = nbs[(target,direction)] ;
    CZ_layers.setdefault(t, []).append((control,target))
##----------------------------------------------------------------------------------------------------------------------
# round 3-5: 
for t in range(3,6):
    for control in Xchecks:
        direction = sX[t] ; target = nbs[(control,direction)]
        CZ_layers.setdefault(t, []).append((control,target))
    for target in Zchecks:
        direction = sZ[t] ; control = nbs[(target,direction)] ;
        CZ_layers.setdefault(t, []).append((control,target))
##----------------------------------------------------------------------------------------------------------------------
t = 6
for control in Xchecks:
    direction = sX[t] ;target = nbs[(control,direction)]
    CZ_layers.setdefault(t, []).append((control,target))
for target in Zchecks:
    direction = sZ[t] ; control = nbs[(target,direction)] ;
    CZ_layers.setdefault(t, []).append((control,target))
##----------------------------------------------------------------------------------------------------------------------
# round 7: 
t = 7
for control in Xchecks:
    direction = sX[t] ; target = nbs[(control,direction)]
    CZ_layers.setdefault(t, []).append((control,target))

# Crosstalk between CZ
from Exp_data.crosstalk_pair import crosstalk_pair
crosstalk_pair = { tuple(convert_item(x) for x in k): [convert_key2(t) for t in v]  for k, v in crosstalk_pair.items() }

with open('Exp_data/crosstalk_result_UI.pkl', 'rb') as f:
    crosstalk_result = pickle.load(f)
crosstalk_result = {  tuple(convert_item(x) for x in k): v for k, v in crosstalk_result.items() }
crosstalk_result = {  k: {'U_I': v['U_I'], 'order': [convert_key([p])[0] for p in v['order'] ] } for k, v in crosstalk_result.items() }

set_crosstalk_gpc = {} ;

basis_states = list(product([0,1,2], repeat = 4))
state_to_index = {state: idx for idx, state in enumerate(basis_states)}

for key, value in crosstalk_result.items():
    arr = crosstalk_result[key]['U_I']
    
    for pair in crosstalk_pair[key]:
        a = f_locate( pair, crosstalk_result[key]["order"] )
        s1 = a[0] ; s2 = a[1] ; 
        
        positions = sorted({0, 1, 2, 3} - {s1, s2})
        state = [0, 0, 0, 0]

        channel_list = [] ;
        for temp in list(product([0,1], repeat = 2)) :
            state[positions[0]] = temp[0] ;  state[positions[1]] = temp[1] 
            indices = []
            
            for temp2 in list(product([0,1,2], repeat = 2)) :
                state[s1] = temp2[0] ; state[s2] = temp2[1] ;
                indices.append(state_to_index[tuple(state)])
            
            sub = arr[np.ix_(indices, indices)]
            channel_list.append( kraus_to_leaky_pauli_channel([sub], 2, 3) )
            # print(kraus_to_leaky_pauli_channel([sub], 2, 3).get_prob_from_to(0, 0, 0) )
        if s1 < s2 :
            index = (crosstalk_result[key]["order"][s1],crosstalk_result[key]["order"][s2])
            set_crosstalk_gpc.setdefault(index, []).append(average_GPC(2, channel_list))
        else:
            index = (crosstalk_result[key]["order"][s2],crosstalk_result[key]["order"][s1])
            set_crosstalk_gpc.setdefault(index, []).append(average_GPC(2, channel_list))

set_crosstalk_gpc = {key: merge_GPC(2, lpc) for key, lpc in set_crosstalk_gpc.items()}


order_crosstalk_gpc = {}
for key, value in set_crosstalk_gpc.items():
    temp = find_key_by_value(crosstalk_pair, key, "CZ")
    order_crosstalk_gpc.setdefault(find_key_by_value(CZ_layers ,temp[0], "CZ"), []).append( key )
# order_crosstalk_gpc


# Crosstalk between CZ and SQ

from Exp_data.crosstalk_pair_SQ import crosstalk_pair_SQ
crosstalk_pair_SQ = { tuple(convert_item(x) for x in k): convert_key2(v)  for k, v in crosstalk_pair_SQ.items() }
with open('Exp_data/crosstalk_result_UI_SQ.pkl', 'rb') as f:
    crosstalk_result_SQ = pickle.load(f)
crosstalk_result_SQ = {  tuple(convert_item(x) for x in k): v for k, v in crosstalk_result_SQ.items() }
crosstalk_result_SQ = {  k: {'U_I': v['U_I'], 'order': \
                             [convert_key([p])[0] for p in v['order'] ] } for k, v in crosstalk_result_SQ.items() }


set_crosstalk_gpc_SQ = {} ;

basis_states = list(product([0,1,2], repeat = 3))
state_to_index = {state: idx for idx, state in enumerate(basis_states)}

for key, value in crosstalk_result_SQ.items():
    arr = crosstalk_result_SQ[key]['U_I']
    
    pair = crosstalk_pair_SQ[key] ;
    a = f_locate( pair, crosstalk_result_SQ[key]["order"] )
    s1 = a[0] ; s2 = a[1] ; 
    
    positions = sorted({0, 1, 2} - {s1, s2})
    state = [0, 0, 0]

    channel_list = [] ;
    for temp in [0, 1] :
        state[positions[0]] = temp ; 
        indices = []
        
        for temp2 in list(product([0,1,2], repeat = 2)) :
            state[s1] = temp2[0] ; state[s2] = temp2[1] ;
            indices.append(state_to_index[tuple(state)])
        
        sub = arr[np.ix_(indices, indices)]
        channel_list.append( kraus_to_leaky_pauli_channel([sub], 2, 3) )
        # print(kraus_to_leaky_pauli_channel([sub], 2, 3).get_prob_from_to(0, 0, 0) )
    if s1 < s2 :
        index = (crosstalk_result_SQ[key]["order"][s1], crosstalk_result_SQ[key]["order"][s2])
        set_crosstalk_gpc_SQ.setdefault(index, []).append(average_GPC(2, channel_list))
    else:
        index = (crosstalk_result_SQ[key]["order"][s2], crosstalk_result_SQ[key]["order"][s1])
        set_crosstalk_gpc_SQ.setdefault(index, []).append(average_GPC(2, channel_list))

set_crosstalk_gpc_SQ = {key: merge_GPC(2, lpc) for key, lpc in set_crosstalk_gpc_SQ.items()}


order_crosstalk_gpc_SQ = {}
for key, value in set_crosstalk_gpc_SQ.items():
    temp = find_key_by_value(crosstalk_pair_SQ, key, "SQ")
    order_crosstalk_gpc_SQ.setdefault(find_key_by_value(CZ_layers ,temp[0], "CZ"), []).append( key )



















