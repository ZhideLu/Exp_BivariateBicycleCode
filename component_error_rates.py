import numpy as np

# # depolarizing noise model 
T1 = 41.8; 
T_CPMG = 39.7 ; T_SpinEcho = 17.4 ; T2 = (T_CPMG + T_SpinEcho) /2 ;

DD_time = 0.92 ;
cycle_time = 1.895 ;

error_rate_init = 0.000
error_rate_idle = 0.0035
error_rate_H = 0.0008
error_rate_cz = 0.0098
error_rate_meas = 0.0403
error_final = 0.0329       # data qubit meas error
error_DD_phaseflip = ( 1 - np.exp(-DD_time/T2) ) / 2
error_DD_bitflip =  ( 1 - np.exp(-DD_time/T1) ) / 2