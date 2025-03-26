# Exp_BivariateBicycleCode



In our work [Demonstration of low-overhead quantum error correction codes], we report the demonstration of two low-overhead quantum low-density parity-check (qLDPC) codes [27], a distance-4 bivariate bicycle code and a distance-3 qLDPC code, on our latest superconducting processor, Kunlun, featuring 32 long-range-coupled transmon qubits. 

Utilizing a two-dimensional architecture with overlapping long-range couplers, we demonstrate simultaneous measurements of all nonlocal weight-6 stabilizers via periodic execution of an efficient syndrome extraction circuit. We achieve a logical error rate per cycle of (31.15 ± 0.51)% for the distance-4 bivariate bicycle code with four logical qubits and (38.45 ± 0.49)% for the distance-3 qLDPC code with six logical qubits. This is on par with running four and six individual patches of distance-4 and distance-3 surface codes, with each patch exhibiting a logical error rate per cycle of 8.91% and 7.77%, respectively. Our results establish the feasibility of implementing various qLDPC codes with long-range coupled superconducting processors, marking a crucial step towards large-scale low-overhead quantum error correction.

Here, we provide the codes for exprimental data analysis and classical simulation and the data for experimental results and simulation results.

## Contents

- [Numerical Simulations](Numerical_Simulations)
- [Experimental Results](Experimental_Results)

## The numerical simulations are built With

* [LDPC](https://github.com/quantumgizmos/ldpc) - A package for decoding quantum error correction codes in Python language
* [bposd](https://pypi.org/project/bposd/) - A decoder for quantum LDPC codes

## Acknowledgments

This project includes substantial portions of code derived from [BivariateBicycleCodes] (https://github.com/sbravyi/BivariateBicycleCodes/tree/main) which is licensed under the Apache License 2.0.