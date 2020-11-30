# This script emulates a 2D turbo spin echo sequence
# Basic, and not real-time.
# Assembler code does not loop but python sends the .txt for each time
# (2D phantom required, there is no slice selection on the 3rd dimension)


import scipy.fft as fft
import scipy.signal as sig

import pdb
import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt
# import scipy.fft as fft
import scipy.signal as sig
import pdb
import math
import time


from local_config import ip_address, port, fpga_clk_freq_MHz
from ocra_lib.assembler import Assembler
import server_comms as sc

from experiment import Experiment

st = pdb.set_trace

def sinc(x, tx_time, Nlobes, alpha):
    y = []
    t0 = (tx_time / 2) / Nlobes
    for ii in x:
        if ii == 0.0:
            yy = 1.0
        else:
            yy = t0 * ((1 - alpha) + alpha * np.cos(ii / Nlobes / t0)) * math.sin(ii / t0) / ii
        y = np.append(y, yy)
    return y

# Experiment parameters
freq_larmor = 2.14769  # local oscillator frequency, MHz
TR = 1e6  # us
TE = 40e3  # us
ETL = 2 # Echo train length
fe_resolution = 64  # number of (I,Q) USEFUL samples to acquire during a shot
pe_step_nr = 2    # number of phase encoding steps
# Delays
sample_nr_dig_filt = 0 # number of additional samples acquired per acquisition for filtering (digital)
# sliceSelWaitEddy = 0 # us

tx_dt = 1  # RF TX sampling dt in microseconds; is rounded to a multiple of clocks (122.88 MHz)
BW = 20000          # Rf Rx Bandwidth
rx_dt = (1 / BW) * 1e6
# rx_dt = 50  # RF RX sampling dt

##### Times have to match with "<instruction_file>.txt" ####
T_tx_Rf = 100       # RF pulse length (us)
T_G_ramp_dur = 250  # Gradient ramp time (us)
# T_G_ramp_Rf_dur = 60  # Gradient ramp time (us)

sample_nr_2_STOP_Seq = 256 + 10000  # Nr. of samples to acquire TO STOP the acquisition

# Correct for DC offset and scaling
scale_G_ss = 0.32
scale_G_pe = 0.32
scale_G_fe = 0.32
offset_G_ss = 0.0
offset_G_pe = 0.0
offset_G_fe = 0.0

# Rf amplitude
Rf_ampl = 0.3 #0.07125  # for Tom
sample_nr_echo = fe_resolution + sample_nr_dig_filt # number of (I,Q) TOTAL samples to acquire during a shot

# Centering the echo
echo_delay1 = 4550  # us; correction for receiver delay
echo_delay2 = 4550  # us; correction for receiver delay

##### RF pulses #####
rx_dt_corr = rx_dt * 0.5  # correction factor to have correct Rx sampling time till the bug is fixed
### 90 RF pulse   ###
# Time vector
t_Rf_90 = np.linspace(0, T_tx_Rf, math.ceil(T_tx_Rf / tx_dt) + 1)  # Actual TX RF pulse length
# sinc pulse
# alpha = 0.46  # alpha=0.46 for Hamming window, alpha=0.5 for Hanning window
# Nlobes = 1
# sinc pulse with Hamming window
# tx90 = Rf_ampl * sinc(math.pi*(t_Rf_90 - T_tx_Rf/2),T_tx_Rf,Nlobes,alpha)
tx90_tight = Rf_ampl * np.ones(np.size(t_Rf_90)) # Hard pulse (square)
tx90 = np.concatenate((tx90_tight, np.zeros(1000 - np.size(tx90_tight))))

### 180 RF pulse ###
tx180 = tx90_tight * 2
tx180 = np.concatenate((tx180, np.zeros(2000 - np.size(tx180) - np.size(tx90))))

##### Gradients #####
t_G_ref_Area_tight = ((1 / ( BW / (fe_resolution))) / 2) * 1e6  # Time for 1/2 K space (square pulse)
t_G_ref_Area_filter = ((1 / (BW / (sample_nr_echo))) / 2) * 1e6  # Time for 1/2 K space with spare samples for filter (square pulse)
# t_G_sliceSel90 = T_tx_Rf + sliceSelWaitEddy * 1e6 + 2*T_G_ramp_Rf_dur # Time for 90 deg slice sel
# t_G_sliceSel180 = t_G_sliceSel90 # Time for refocusing (180 deg) slice sel
T_G_pe_dur = t_G_ref_Area_tight + T_G_ramp_dur  # Total phase encoding gradient ON time length (us)
T_G_pre_fe_dur = t_G_ref_Area_filter + (3/2)*T_G_ramp_dur  # Total freq. encoding REWINDER ON time length (us)
T_G_fe_dur = 2 * (t_G_ref_Area_filter + T_G_ramp_dur)  # Total Frequency encoding gradient ON time length (us)

# # Rf slice selection encoding gradient shapes (while 90 and refocusing pulses (180))
# grad_ss90_samp_nr = math.ceil(t_G_sliceSel90 / 10)
# grad_ramp_Rf_samp_nr = math.ceil(T_G_ramp_Rf_dur / 10)
# grad_ss90 = np.hstack([np.linspace(0, 1, grad_ramp_Rf_samp_nr+1)[:-1],  # Ramp up +
#                        np.ones(int(T_tx_Rf/10)),  # Top +
#                        np.linspace(1, 0, grad_ramp_Rf_samp_nr+1)[:-1], # Ramp down +
#                        np.linspace(0, -1, grad_ramp_Rf_samp_nr+1)[:-1], # Ramp down -
#                        (np.ones(np.ceil((T_tx_Rf/10)/2 - grad_ramp_Rf_samp_nr/2).astype(int))*-1),  # Top -
#                        np.linspace(-1, 0, grad_ramp_Rf_samp_nr+1)[:-1]])  # Ramp up -
# # plt.plot(np.arange(len(grad_ss90))*10,np.cumsum(grad_ss90), 'b.-')
# # plt.plot(np.arange(len(grad_ss90))*10,grad_ss90, 'r.-')
# # plt.grid()
#
# grad_ss180_samp_nr = math.ceil(t_G_sliceSel180 / 10)
# grad_ss180 = np.hstack([np.linspace(0, 1, grad_ramp_Rf_samp_nr+1)[:-1],  # Ramp up +
#                        np.ones(int(T_tx_Rf/10)),  # Top +
#                        np.linspace(1, 0, grad_ramp_Rf_samp_nr+1)[:-1]]) # Ramp down +

# Phase encoding gradient shape
grad_ramp_samp_nr = math.ceil(T_G_ramp_dur / 10)
grad_pe_samp_nr = math.ceil(T_G_pe_dur / 10)
grad_pe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                     np.ones(grad_pe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                     np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
grad_pe = np.hstack([grad_pe, np.zeros(190 - np.size(grad_pe))])

# Pre-frequency encoding gradient shape
grad_pre_fe_samp_nr = math.ceil(T_G_pre_fe_dur / 10)
grad_pre_fe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                         np.ones(grad_pre_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                         np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
grad_pre_fe = np.hstack([grad_pre_fe, np.zeros(200 - np.size(grad_pre_fe))])

# Frequency encoding gradient shape
grad_fe_samp_nr = math.ceil(T_G_fe_dur / 10)
grad_fe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                     np.ones(grad_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                     np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
sample_nr_center_G_fe = (((1 / (BW / 140)) / 2) * 1e6 + T_G_ramp_dur)/10 # Total phase encoding gradient ON time length (us)
grad_fe = np.hstack([grad_fe, np.zeros(np.round(380 - np.size(grad_fe)).astype('int'))])

# Initialisation of the DAC
exp = Experiment(samples=4,  # number of (I,Q) samples to acquire during a shot of the experiment
                 lo_freq=freq_larmor,  # local oscillator frequency, MHz
                 tx_t=tx_dt,
                 instruction_file="ocra_lib/se_default_vn.txt")
exp.initialize_DAC()
time.sleep(1)

# Arrange kSpace filling
interleaved = 0
scale_G_pe_range = np.linspace(-1, 1, pe_step_nr)
# to acquire non-consecutive kSpace lines
if interleaved == 0:
    TR_nr = np.ceil(pe_step_nr / ETL).astype(int)
    kIdxTmp = np.zeros([TR_nr,ETL]).astype(int)
    kIdxTmp2 = np.zeros([TR_nr,ETL]).astype(int)
    for idx3 in range(np.ceil(TR_nr/2).astype(int)):
        kIdxTmp[idx3,:] = idx3+(np.linspace(0, ETL-1, ETL).astype(int))*np.ceil(TR_nr/2).astype(int)
    # Reorder here the echoes
    kIdxTmp[np.ceil(TR_nr / 2).astype(int):, :] = (kIdxTmp[:(np.ceil(TR_nr / 2).astype(int)), :] + 1) * -1
    kIdxTmp2[0::2, :] = kIdxTmp[:np.ceil(TR_nr / 2).astype(int), :]
    kIdxTmp2[1::2, :] = kIdxTmp[np.ceil(TR_nr / 2).astype(int):, :]

# Loop repeating TR and updating the gradients waveforms
data = np.zeros([sample_nr_2_STOP_Seq, TR_nr], dtype=complex)

for idxTR in range(TR_nr):
    exp = Experiment(samples=sample_nr_2_STOP_Seq,  # number of (I,Q) samples to acquire during a shot of the experiment
                     lo_freq=freq_larmor,  # local oscillator frequency, MHz
                     tx_t=tx_dt,           # RF TX sampling time in microseconds; will be rounded to a multiple of system clocks (122.88 MHz)
                     rx_t=rx_dt_corr,  # RF RX sampling time in microseconds; as above
                     instruction_file="TSE_2D_tests.txt") # TSE_2D_tests.txt
    ###### Send waveforms to RP memory ###########
    tx_length = np.zeros(1).astype(int)
    # Load the RF waveforms
    tx_idx = exp.add_tx(tx90.astype(complex))               # add 90x+ Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx90.size])
    tx_idx = exp.add_tx(tx180.astype(complex))              # add 180x+ Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx180.size])
    tx_idx = exp.add_tx(tx180.astype(complex)*1j)           # add 180y+ Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx180.size])
    tx_idx = exp.add_tx(tx180.astype(complex)*(-1j))        # add 180y- Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx180.size])

    # Adjust gradient waveforms
    # Echo nr                  |               1               |               2               |
    # Block    |    1   |   2  |  3  |   4   |   5     |   6   |  7  |   8   |    9    |  10   |
    # Mem      |    0   |   2  |  3  |   4   |   5     |   6   |  7  |   8   |    9    |  10   |
    # RF       |_$$_____|______|_$$__|_______|_________|_______|_$$__|_______|_________|_______|_$$_
    #          |        |      |     |       |         |       |     |       |         |       |
    # Gss      |/--\   _|______|/--\_|_______|_________|_______|/--\_|_______|_________|_______|/--\
    #          |    \_/ |      |     |       |         |       |     |       |         |       |
    # Gpe      |________|______|_____|/1111\_|_________|      _|_____|/2222\_|_________|      _|____
    #          |        |      |     |       |         |\1111/ |     |       |         |\2222/ |
    # Gfe      |________|/---\_|_____|_______|/------\_|_______|_____|_______|/------\_|_______|____


    # Shape Gradients block by block
    G_length = np.zeros(1).astype(int)
    # Block 1: Rf90 + ss
    # Block 2: -fe/2 prephase
    grad_fe_2_corr = grad_pre_fe * scale_G_fe + offset_G_fe
    grad_pe_2_corr = np.zeros(np.size(grad_fe_2_corr)) + offset_G_pe
    grad_ss_2_corr = np.zeros(np.size(grad_fe_2_corr)) + offset_G_ss
    grad_idx = exp.add_grad(grad_ss_2_corr, grad_pe_2_corr, grad_fe_2_corr)
    G_length = np.hstack([G_length, G_length[-1] + grad_fe_2_corr.size])

    for idxETL in range(ETL):
        scale_G_pe_sweep = scale_G_pe_range[kIdxTmp2[idxTR, idxETL]]
        # ----echo 1--------------------
        # Block 3: Rf180 + ss
        # Block 4: pe+
        grad_pe_4_corr = grad_pe * scale_G_pe * scale_G_pe_sweep + offset_G_pe
        grad_fe_4_corr = np.zeros(np.size(grad_pe_4_corr)) + offset_G_fe
        grad_ss_4_corr = np.zeros(np.size(grad_pe_4_corr)) + offset_G_ss
        grad_idx = exp.add_grad(grad_ss_4_corr, grad_pe_4_corr, grad_fe_4_corr)
        G_length = np.hstack([G_length, G_length[-1] + grad_fe_4_corr.size])

        # Block 5: fe
        grad_fe_5_corr = grad_fe * scale_G_fe + offset_G_fe
        grad_pe_5_corr = np.zeros(np.size(grad_fe_5_corr)) + offset_G_pe
        grad_ss_5_corr = np.zeros(np.size(grad_fe_5_corr)) + offset_G_ss
        grad_idx = exp.add_grad(grad_ss_5_corr, grad_pe_5_corr, grad_fe_5_corr)
        G_length = np.hstack([G_length, G_length[-1] + grad_fe_5_corr.size])

        # Block 6: pe-
        grad_pe_6_corr = grad_pe * (-scale_G_pe) * scale_G_pe_sweep + offset_G_pe
        grad_fe_6_corr = np.zeros(np.size(grad_pe_6_corr)) + offset_G_fe
        grad_ss_6_corr = np.zeros(np.size(grad_pe_6_corr)) + offset_G_ss
        grad_idx = exp.add_grad(grad_ss_6_corr, grad_pe_6_corr, grad_fe_6_corr)
        G_length = np.hstack([G_length, G_length[-1] + grad_fe_6_corr.size])

    # Run command to MaRCoS
    data[:, idxTR] = exp.run()

# time vector for representing the received data
samples_data = len(data)
t_rx = np.linspace(0, rx_dt * samples_data, samples_data)  # us

plt.figure(1)
plt.subplot(2,1,1)
# plt.plot(t_rx, np.real(data))
# plt.plot(t_rx, np.abs(data))
plt.plot(np.real(data))
plt.plot(np.abs(data))
plt.legend(['real', 'abs'])
plt.xlabel('time (us)')
plt.ylabel('signal received (V)')
plt.title('Total sampled data = %i' % samples_data)
plt.grid()

echo_shift_idx_1 = np.floor(echo_delay1 / rx_dt).astype('int')
echo_shift_idx_2 = np.floor(echo_delay2 / rx_dt).astype('int')
kspace = np.zeros([sample_nr_echo, pe_step_nr]).astype(complex)
kspace[:, 0::2] = data[echo_shift_idx_1:echo_shift_idx_1 + sample_nr_echo, :]
kspace[:, 1::2] = data[echo_shift_idx_2:echo_shift_idx_2 + sample_nr_echo, :]

# timestr = time.strftime("%Y%m%d-%H%M%S")
# filemane = timestr + str("outfile")
# np.savez(filemane, data=data, t_rx=t_rx, kspace=kspace)

plt.subplot(2, 1, 2)
plt.plot(np.real(kspace))
plt.plot(np.abs(kspace))
plt.legend(['real', 'abs'])
plt.xlabel('Sample nr.')
plt.ylabel('signal received (V)')
plt.title('Echo time in acquisition from = %f' % t_rx[echo_shift_idx_1])
plt.grid()

plt.figure(2)
plt.subplot(1, 2, 1)
Y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspace)))
img = np.abs(Y)
plt.imshow(np.abs(kspace), cmap='gray')
plt.title('k-Space')
plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.title('image')
plt.show()
