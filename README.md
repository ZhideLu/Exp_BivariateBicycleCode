# Exp_BivariateBicycleCode



In our work [Demonstration of low-overhead quantum error correction codes], we report the demonstration of two low-overhead quantum low-density parity-check (qLDPC) codes [27], a distance-4 bivariate bicycle code and a distance-3 punctured bivariate bicycle code code, on our latest superconducting processor, Kunlun, featuring 32 long-range-coupled transmon qubits. 

Utilizing a two-dimensional architecture with overlapping long-range couplers, we demonstrate simultaneous measurements of all nonlocal weight-6 stabilizers via periodic execution of an efficient syndrome extraction circuit. We achieve a logical error rate per logical qubit per cycle of (8.91 ± 0.17)% for the distance-4 bivariate bicycle code with four logical qubits and (7.77 ± 0.12)% for the distance-3 punctured bivariate bicycle code with six logical qubits.  Our results establish the feasibility of implementing various qLDPC codes with long-range coupled superconducting processors, marking a crucial step towards large-scale low-overhead quantum error correction.

Here, we provide the codes for exprimental data analysis and classical simulation and the data for experimental results and simulation results.

## The numerical simulations are built With

* [Python](https://www.python.org/) - Version 3.10.14 ;
* [Stim](https://github.com/quantumlib/Stim/tree/main) -Version 1.13.0, a tool for high performance simulation and analysis of  quantum error correction circuits ;
* [LDPC](https://github.com/quantumgizmos/ldpc) - Version 0.1.53, a package for decoding quantum error correction codes in Python language ;
* [bposd](https://pypi.org/project/bposd/) - Version 1.6, a decoder for quantum LDPC codes ;
* [leaky](https://github.com/inmzhang/leaky) - Version 1.6, an implementation of Google's Pauli+ simulator ;

## Acknowledgments

This project includes substantial portions of code derived from [BivariateBicycleCodes] (https://github.com/sbravyi/BivariateBicycleCodes/tree/main) which is licensed under the Apache License 2.0.