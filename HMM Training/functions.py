from python_speech_features import mfcc
from python_speech_features import delta
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from numpy import genfromtxt
import math
import scipy.fftpack

def logAdd(a, b):
    l0 = -1 * math.pow(10, 30)
    if a > b:
        # We want a < b; swap if it's not true
        oldB = b
        b = a
        a = oldB
    d = a - b
    if d < -1 * math.log(-1 * l0):
        return b
    else:
        return b + math.log(1 + math.exp(a - b))

def MFCC(LOCATION, SAVELOCATION):
    # NOTE: Currently using a library for generating MFCCs, to make sure that
    # the implementation of the EM algorithm is based on correct values. Own
    # implementation of MFCCs can be found commented out below.
    data = genfromtxt(LOCATION, delimiter = ",")
    signal = np.zeros([64000, 1])
    for i in range(0, 64000):
        signal[i, 0] = data[i]

    mfcc_feat = np.transpose(mfcc(signal, 16000))
    d_mfcc_feat = (delta(mfcc_feat, 2))

    rows, columns = np.shape(mfcc_feat)

    # Concatenate matrices
    # MFCC matrix is on top of the delta matrix
    mfcc_deltas = np.zeros([rows * 2, columns])
    for i in range(0, rows):
        mfcc_deltas[i, :] = mfcc_feat[i, :]
        mfcc_deltas[i + rows, :] = d_mfcc_feat[i, :]
    np.savetxt(SAVELOCATION, mfcc_deltas, delimiter = ",")
    return mfcc_deltas
        
    # # MFCC and Delta feature code
    # SKIP          = 160                      # Numbers of samples to skip to start new frame
    # SAMPLES       = 400                      # Number of samples per frame
    # LOWER_FREQ    = 300.0                    # Lower frequency in Hz
    # UPPER_FREQ    = 8000.0                   # Upper frequency in Hz
    # NFILTERS      = 26                       # Number of filterbanks for
    # Fs            = 16000                    # Sampling frequency in Hz
    # FRAME_LEN     = 0.025                    # Length of each frame in seconds
    # AUDIO_LEN     = 4                        # Length of audio sample in seconds
    # M             = 2                        # Window size of delta features
    # NCOEFFICIENTS = 13                       # Number of MFCC coefficients to keep
    #
    # next_power = 1
    # my_pad = int(np.power(2, (next_power - 1) + np.ceil(np.log2(SAMPLES))))
    # FFT_LENGTH    = my_pad                   # Length of FFT
    # FFT_KEEP      = FFT_LENGTH // 2 + 1      # Number of FFT coefficients to keep
    #
    # # Import data
    # raw = genfromtxt(LOCATION, delimiter = ",")
    #
    # # NOTE: ROWS is actually columns. Need to fix.
    # # Perform the FFT and create the power spectrum
    # ROWS = (Fs * AUDIO_LEN - SAMPLES) // SKIP + 1
    # power_spec = np.zeros([ROWS, FFT_LENGTH])
    # index = 0
    # for i in range(0, len(raw) - SAMPLES, SKIP):
    #     vector = []
    #     for j in range(i, i + SAMPLES):
    #         vector.append(raw[j])
    #     power_spec[index, :] = np.multiply((np.power(np.absolute(np.fft.fft(vector, n = FFT_LENGTH)), 2)), 0.0025)
    #     index = index + 1
    #
    # # Trim the power spectrum
    # trimmed_power_spec = np.zeros([ROWS, FFT_KEEP])
    # for i in range(0, ROWS):
    #     trimmed_power_spec[i, :] = power_spec[i, 0:FFT_KEEP]
    #
    # # Find Mels of lower and upper frequencies
    # LOWER_MEL = 1125.0 * math.log(1.0 + (LOWER_FREQ / 700.0))
    # UPPER_MEL = 1125.0 * math.log(1.0 + (UPPER_FREQ / 700.0))
    #
    # # Create the filterbanks in Mels
    # mel_banks = np.linspace(LOWER_MEL, UPPER_MEL, num = NFILTERS + 2)
    #
    # # Convert the filterbanks back to frequency
    # freq_banks = [];
    # for i in range(0, len(mel_banks)):
    #     freq_banks.append(700.0 * (math.exp(mel_banks[i] / 1125.0) - 1.0))
    #
    # bins = []
    # for i in range(0, len(freq_banks)):
    #     bins.append(math.floor((FFT_KEEP + 1.0) * freq_banks[i] / 8000))
    #
    # # Make the Mel filterbank
    # filterbank = np.zeros([NFILTERS, FFT_KEEP])
    # for i in range(0, NFILTERS):
    #     # For each filter in the filterbank
    #     for j in range(int(bins[i]), int(bins[i + 1])):
    #         filterbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
    #     for j in range(int(bins[i + 1]), int(bins[i + 2])):
    #         filterbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])
    #
    # # Collect coefficients/filterbank energies
    # energies = np.zeros([ROWS, NFILTERS])
    # for i in range(0, ROWS):
    #     # For each frame
    #     for j in range(0, NFILTERS):
    #         # For each filter in the filterbank
    #         energies[i, j] = np.dot(trimmed_power_spec[i, :], np.transpose(filterbank[j, :]))
    #
    # # Take the natural log and then the inverse Fourier transform of the energies
    # log_energies = np.log(energies)
    # ifft_log_energies = scipy.fftpack.dct(log_energies)
    #
    # # Trim to keep only the first 13 energies
    # # Save the matrix in a 13 x T matrix
    # mfcc = np.transpose(ifft_log_energies[:, 0:NCOEFFICIENTS])
    #
    # # Find deltas
    # deltas = np.zeros([NCOEFFICIENTS, ROWS - 2 * M])
    # for i in range(0, NCOEFFICIENTS):
    #     # For each row in the MFCC matrix
    #     for j in range(0, ROWS - 2 * M):
    #         # Calculate summation to find delta coefficient
    #         d = 0
    #         for n in range(1, M + 1):
    #             d = d + ((n * (mfcc[i, j + 2 * n] - mfcc[i, j])) / (2 * (n ** 2)))
    #         deltas[i, j] = d
    #
    # # Trim the original MFCC matrix
    # cut_mfcc = np.zeros([NCOEFFICIENTS, ROWS - 2 * M])
    # for i in range(0, NCOEFFICIENTS):
    #     cut_mfcc[i, :] = mfcc[i, 0:ROWS - 2 * M]
    #
    # # Concatenate the original MFCC matrix with the delta matrix
    # # The MFCC matrix is above the delta matrix
    # final = np.zeros([NCOEFFICIENTS * 2, ROWS - 2 * M])
    # for i in range(0, NCOEFFICIENTS):
    #     final[i, :] = cut_mfcc[i, :]
    #     final[i + NCOEFFICIENTS, :] = deltas[i, :]
    # np.savetxt(SAVELOCATION, final, delimiter = ",")
    # return final
