from typing import List, Sequence, Tuple
import itertools
import functools
import numpy as np

from collections import defaultdict
from itertools import permutations
from tabulate import tabulate

from scipy.io import savemat
from scipy.io import loadmat

import leaky
from leaky import LeakyPauliChannel
from leaky.utils import (
    leakage_status_tuple_to_int,
    _l2p,
    _get_projector_slice,
    _project_kraus_with_initial_final,
    _scatter_status,
    PAULIS,
    TWO_QUBITS_PAULIS,
    LeakageStatus,
    ProjectStatus,
)

import os
import sys

## decompose kraus operators into GPC
def kraus_to_leaky_pauli_channel(
    kraus_operators: Sequence[np.ndarray],
    num_qubits: int,
    num_level: int,
    safety_check: bool = True,
) -> LeakyPauliChannel:

    if num_qubits not in [1, 2]:
        raise ValueError("Only 1 or 2 qubits operators are supported.")
    list_channel = [] ;
    
    all_status = list(itertools.product(range(num_level - 1), repeat=num_qubits))
    
    for kraus in kraus_operators:
        for initial_status, final_status in itertools.product(all_status, repeat=2):
            num_u = sum(
                s0 == 0 and s1 > 0 for s0, s1 in zip(initial_status, final_status)
            )
            q_in_r = [
                i
                for i, (s0, s1) in enumerate(zip(initial_status, final_status))
                if s0 == 0 and s1 == 0
            ]
            num_r = len(q_in_r)

            prefactor: float = 1.0 / 2**num_u
            projectors = _scatter_status(initial_status, final_status)
            
            for initial_projector, final_projector in projectors:
                projected_kraus = _project_kraus_with_initial_final(
                    kraus, num_qubits, num_level, initial_projector, final_projector)
                probability: float
                pauli_channel: List[Tuple[int, float]] = []
                
                if num_r == 0:
                    assert projected_kraus.shape == (1, 1)
                    probability = (
                        prefactor * np.abs(projected_kraus).astype(float) ** 2
                    ).item()
                    pauli_channel.append((0, probability))
                else:
                    dim = 2**num_r
                    assert projected_kraus.shape == (dim, dim)
                    for i, paulis in enumerate(itertools.product(PAULIS, repeat=num_r)):
                        probability = (
                            prefactor
                            * np.abs(
                                np.trace(
                                    projected_kraus @ functools.reduce(np.kron, paulis)
                                )
                                / dim
                            )
                            ** 2
                        )
                        idx = sum(
                            [
                                ((i >> (2 * (num_r - j - 1))) & 0b11)
                                << (2 * (num_qubits - q - 1))
                                for j, q in enumerate(q_in_r)
                            ] )
                        pauli_channel.append((idx, probability))

                    probability = sum([p for _, p in pauli_channel])
                for idx, p in pauli_channel:
                    list_channel.append([leakage_status_tuple_to_int(initial_status),
                                     leakage_status_tuple_to_int(final_status),
                                     idx,
                                     p
                                    ])
    merged = defaultdict(float)
    for i, j, k, val in list_channel:
        merged[(i, j, k)] += val
    
    p_unique = [[i, j, k, v] for (i, j, k), v in merged.items()]
    pp_sum = [np.sum([t[3] for t in p_unique if t[0]==i ] ) for i in [0,1,2, 16,17,18, 32,33,34] ]
    for entry in p_unique:
        index = (entry[0]//16)*3+(entry[0]%16)
        if pp_sum[index] > 0:
            entry[3] /= pp_sum[index] 
    
    channel = LeakyPauliChannel(is_single_qubit_channel=num_qubits == 1)
    
    for item in p_unique:
        channel.add_transition(item[0], item[1], item[2], item[3],)
        
    if safety_check:
        channel.safety_check()

    return channel
