# This script acquires a 2D spin echo sequence 
# Basic, and not real-time.
# Assembler code does not loop but phython sends the .txt for each time
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

st = pdb.set_trace

from local_config import ip_address, port, fpga_clk_freq_MHz
from ocra_lib.assembler import Assembler
import server_comms as sc

from experiment import Experiment


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

def se_2D_v0_RP():
    sample_nr = 256+1000             # number of (I,Q) samples to acquire during a shot of the experiment
    pe_step_nr = 256             # number of phase encoding steps
    tx_dt = 0.1                 # RF TX sampling time in microseconds; will be rounded to a multiple of system clocks (122.88 MHz)
    rx_dt = 50                 # desired sampling dt
    rx_dt_corr = rx_dt * 0.5    # correction factor till the bug is fixed
    sample_nr_echo = 256
    ##### Times in "<instruction_file>.txt" #####
    TE = 0.001e6  # Echo time (us)
    T_tx_Rf = 100  # RF pulse length (us)
    T_G_ramp_dur = 250  # Gradient ramp time
    BW = 20000  # Gradient ramp time
    T_G_pe_dur = ((1/(BW/sample_nr_echo))/2)*1e6 + T_G_ramp_dur # Total phase encoding gradient ON time length (us)
    T_G_pre_fe_dur = ((1/(BW/sample_nr_echo))/2)*1e6+2*T_G_ramp_dur  # Total phase encoding gradient ON time length (us)
    T_G_fe_dur = 2 * (((1/(BW/sample_nr_echo))/2) * 1e6) + (2 * T_G_ramp_dur)  # Frequency encoding gradient ON time length (us)

    ##### RF pulses #####
    ### 90 RF pulse   ###
    # Time vector
    t_Rf_90 = np.linspace(0, T_tx_Rf, math.ceil(T_tx_Rf / tx_dt) + 1)  # Actual TX RF pulse length
    # sinc pulse
    alpha = 0.46  # alpha=0.46 for Hamming window, alpha=0.5 for Hanning window
    Nlobes = 1
    Rf_ampl = 0.125*0.57
    # sinc pulse with Hamming window
    # tx90 = Rf_ampl * sinc(math.pi*(t_Rf_90 - T_tx_Rf/2),T_tx_Rf,Nlobes,alpha)
    tx90 = Rf_ampl * np.ones(np.size(t_Rf_90))
    ### 180 RF pulse ###
    # sinc pulse     
    tx180 = tx90 * 2
    tx90 = np.concatenate((tx90, np.zeros(2000 - np.size(tx90))))
    tx180 =np.concatenate((tx180, np.zeros(4000-np.size(tx180)-np.size(tx90))))

    # For testing ONLY: echo centering
    # acq_shift = 0
    # tx90_echo_cent = np.hstack((
    #     np.zeros(np.floor(T_G_fe_dur/(2*tx_dt)).astype('int')- np.floor(np.size(tx90)/2).astype('int')-acq_shift),
    #     tx90 ,
    #     np.zeros(np.floor(T_G_fe_dur/(2*tx_dt)).astype('int')- np.floor(np.size(tx90)/2).astype('int')+acq_shift)))
    # tx90_echo_cent = tx90_echo_cent * 0
    ##### Gradients #####
    # Phase encoding gradient shape
    grad_pe_samp_nr = math.ceil(T_G_pe_dur / 10)
    grad_ramp_samp_nr = math.ceil(T_G_ramp_dur / 10)
    grad_pe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                         np.ones(grad_pe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                         np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
    grad_pe = np.hstack([grad_pe, np.zeros(700 - np.size(grad_pe))])
    # Pre-frequency encoding gradient shape
    grad_pre_fe_samp_nr = math.ceil(T_G_pre_fe_dur / 10)
    grad_pre_fe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                         np.ones(grad_pre_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                         np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
    grad_pre_fe = np.hstack([grad_pre_fe, np.zeros(700 - np.size(grad_pre_fe))])
    # Frequency encoding gradient shape
    grad_fe_samp_nr = math.ceil(T_G_fe_dur / 10)
    grad_fe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                         np.ones(grad_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                         np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
    grad_fe = np.hstack([grad_fe, np.zeros(1400 - np.size(grad_fe))])
    # Correct for DC offset and scaling
    scale_G_x = 0.32
    scale_G_y = 0.32
    scale_G_z = 0.32
    offset_G_x = 0.05
    offset_G_y = 0.05
    offset_G_z = 0.0

    # # Loop repeating TR and updating the gradients waveforms
    data = np.zeros([sample_nr, pe_step_nr], dtype=complex)
    # exp = Experiment(samples=sample_nr,  # number of (I,Q) samples to acquire during a shot of the experiment
    #                  lo_freq=2.147,  # local oscillator frequency, MHz
    #                  tx_t=tx_dt,
    #                  # RF TX sampling time in microseconds; will be rounded to a multiple of system clocks (122.88 MHz)
    #                  rx_t=rx_dt_corr,  # RF RX sampling time in microseconds; as above
    #                  instruction_file="ocra_lib/se_default_vn.txt")
    # exp.initialize_DAC()

    for idx2 in range(pe_step_nr):
        exp = Experiment(samples=sample_nr,  # number of (I,Q) samples to acquire during a shot of the experiment
                             lo_freq=2.147,  # local oscillator frequency, MHz
                         tx_t=tx_dt,
                         # RF TX sampling time in microseconds; will be rounded to a multiple of system clocks (122.88 MHz)
                         rx_t=rx_dt_corr,  # RF RX sampling time in microseconds; as above
                         instruction_file="se_2D_LUMC_v0.txt")
        ###### Send waveforms to RP memory ###########
        # Load the RF waveforms
        tx_idx = exp.add_tx(tx90)  # add the data to the ocra TX memory
        tx_idx = exp.add_tx(tx180)  # add the data to the ocra TX memory
        # tx_idx = exp.add_tx(tx90_echo_cent)
        scale_G_pe_sweep = 2 * (idx2 / (pe_step_nr - 1) - 0.5)

        grad_x_1_corr = grad_pre_fe * scale_G_x + offset_G_x
        grad_y_1_corr = grad_pe * scale_G_y * scale_G_pe_sweep + offset_G_y
        grad_z_1_corr = np.zeros(np.size(grad_x_1_corr))+ offset_G_z
        grad_x_2_corr = grad_fe * scale_G_x + offset_G_x
        grad_y_2_corr = np.zeros(np.size(grad_x_2_corr)) + offset_G_y
        grad_z_2_corr = np.zeros(np.size(grad_x_2_corr)) + offset_G_z

        l_tmp = 1345 # 1345
        grad_idx = exp.add_grad(grad_x_1_corr, grad_y_1_corr, grad_z_1_corr)
        grad_idx = exp.add_grad(grad_x_2_corr[0:l_tmp] , grad_y_2_corr[0:l_tmp] , grad_z_2_corr[0:l_tmp])
        # grad_idx = exp.add_grad(grad_x_pe_corr[0:l_tmp], grad_y_pe_corr[0:l_tmp], grad_z_pe_corr[0:l_tmp])
        # grad_idx = exp.add_grad(np.zeros(np.size(grad_fe_corr)), grad_fe_corr, np.zeros(np.size(grad_fe_corr)))
        data[:, idx2] = exp.run()
        # data = exp.run()
        data_mV = data * 1000    # data = np.zeros([sample_nr, pe_step_nr])

    # time vector for representing the received data
    samples_data = len(data)
    t_rx = np.linspace(0, rx_dt * samples_data, samples_data)  # us

    plt.plot(t_rx, np.real(data_mV))
    # plt.plot(t_rx, np.imag(data_mV))
    plt.plot(t_rx, np.abs(data_mV))
    plt.legend(['real', 'imag', 'abs'])
    plt.xlabel('time (us)')
    plt.ylabel('signal received (mV)')
    plt.title('sampled data = %i' % samples_data)
    plt.grid()

    echo_delay = 96.5 # ms
    echo_shift_idx = np.floor(echo_delay/rx_dt).astype('int')
    echo_idx = np.floor(T_G_fe_dur / (2 * rx_dt)).astype('int') - np.floor(sample_nr_echo/ 2).astype('int')
    kspace = data[echo_idx+echo_shift_idx:echo_idx+echo_shift_idx+sample_nr_echo, : ]
    # Y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspace)))

    plt.figure(2)
    plt.plot(t_rx[echo_idx+echo_shift_idx:echo_idx+echo_shift_idx+sample_nr_echo], np.real(kspace))
    # plt.plot(t_rx, np.imag(data_mV))
    plt.plot(t_rx[echo_idx+echo_shift_idx:echo_idx+echo_shift_idx+sample_nr_echo], np.abs(kspace))
    plt.legend(['real', 'imag', 'abs'])
    plt.xlabel('time (us)')
    plt.ylabel('signal received (mV)')
    plt.title('sampled data = %i' % samples_data)
    plt.grid()

    Y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspace)))
    img = np.abs(Y)
    plt.figure(3)
    plt.imshow(img, cmap='gray')
    plt.title('image')
    plt.show()

if __name__ == "__main__":
    se_2D_v0_RP()
