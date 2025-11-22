# Exp_BivariateBicycleCode
In our work [Demonstration of low-overhead quantum error correction codes](https://arxiv.org/abs/2505.09684), we report the demonstration of two low-overhead quantum low-density parity-check (qLDPC) codes, a distance-4 bivariate bicycle code and a distance-3 punctured bivariate bicycle code code, on our latest superconducting processor, Kunlun, featuring 32 long-range-coupled transmon qubits. 

Utilizing a two-dimensional architecture with overlapping long-range couplers, we demonstrate simultaneous measurements of all nonlocal weight-6 stabilizers via periodic execution of an efficient syndrome extraction circuit. We achieve a logical error rate per logical qubit per cycle of (8.91 ± 0.17)% for the distance-4 bivariate bicycle code with four logical qubits and (7.77 ± 0.12)% for the distance-3 punctured bivariate bicycle code with six logical qubits.  Our results establish the feasibility of implementing various qLDPC codes with long-range coupled superconducting processors, marking a crucial step towards large-scale low-overhead quantum error correction.

Here, we provide the codes for exprimental data analysis and classical simulation and the data for experimental results and simulation results.

## The numerical simulations are built With
* [Python](https://www.python.org/) - Version 3.10.14 ;
* [Stim](https://github.com/quantumlib/Stim/tree/main) -Version 1.13.0, a tool for high performance simulation and analysis of  quantum error correction circuits ;
* [LDPC](https://github.com/quantumgizmos/ldpc) - Version 0.1.53, a package for decoding quantum error correction codes in Python language ;
* [bposd](https://pypi.org/project/bposd/) - Version 1.6, a decoder for quantum LDPC codes ;
* [leaky](https://github.com/inmzhang/leaky) - Version 0.2.2, an implementation of Google's Pauli+ simulator ;


## Key folders/files and descriptions
- `functions_BB_code.py`  
  contains functions used in both experimental data analysis and classical simulation.

- `component_error_rates.py`  
  contains physical error rate data for quantum operations on our Kunlun processor.

- `18_4_4/Experimental_data/`  
  **experimental data for the [[18,4,4]] code**

- `18_6_3/Experimental_data/` 
  **experimental data for the [[18,6,3]] code**

- `18_4_4/Pauli_plus_simulation/` 
  Pauli+ simualtion and error budget analysis of the [[18,4,4]] code

- `18_6_3/Pauli_plus_sim/` 
  Pauli+ simualtion of the [[18,6,3]] code

- `Performance_projection/` 
  Pauli simualtion for BB code performance projection


##  Reproducing experimental results
To reproduce experimental results:
- Run: 

  - `18_4_4/Plot_figures_experiment_18_4_4.ipynb`  

  - `18_6_3/Plot_figures_experiment_18_6_3.ipynb`

These notebooks:

- Decode the experimental measured error syndromes  
- Plot:
  - **Fig.3a,b**
  - **Extended Data Fig.3**
  - **Fig.2b**, **Extended Data Fig.2**

We note that this decoding step takes several hours to complete. If you wish to speed up the decoding, you may open the file "18_4_4/ErrorCorrection_for_experiment_18_4_4.py" or "18_6_3/ErrorCorrection_for_experiment_18_6_3.py" and reduce the value of "my_max_iter = 10000" to a smaller number.

## Pauli+ simulation 

To simulate the experiment using the Pauli+ framework:

- Run:
  - `18_4_4/Pauli_plus_simulation/Run_PauliPlus.ipynb`
  - `18_6_3/Pauli_plus_sim/run_PauliPlus.ipynb`

These notebooks produce the Pauli+ simulation results in:

  - **Fig.2c**, **Extended Data Fig.2**
  - **Fig.3a,b** 
  - **Extended Data Fig.3**

To conduct error budget analysis:

- Run:
  - `18_4_4/Pauli_plus_simulation/Logical_error_budget/main_logical_error_budget.ipynb`
  - `18_4_4/Pauli_plus_simulation/Detection_error_budget/main_detect_budget.ipynb`

These notebooks produce the results in:

  - **Fig.2d**,
  - **Fig.3c** 

## Performance Projection
To estimate the logical performance of BB codes in future devices with improved performance, we simulate the BB codes with distances ranging from 4 to 12 under varying physical error rates. For efficiency, only Pauli errors are included in the simulation.

- Run:
  - `Performance_projection/18_4_4/run_18_4_4.ipynb`
  - `Performance_projection/36_4_6/run_36_4_6.ipynb`
  - `Performance_projection/54_4_8/run_54_4_8.ipynb`
  - `Performance_projection/90_8_10/run_90_8_10.ipynb`
  - `Performance_projection/144_12_12/run_144_12_12.ipynb`

- Run:
  - `Performance_projection/Prediction.ipynb`

These notebooks produce the results in:

- **Fig.3c,d **

## Acknowledgments
This project includes substantial portions of code derived from [BivariateBicycleCodes] (https://github.com/sbravyi/BivariateBicycleCodes/tree/main) which is licensed under the Apache License 2.0.