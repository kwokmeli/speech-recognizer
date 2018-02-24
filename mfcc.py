import pyaudio
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from numpy import genfromtxt
import wave
import math
import scipy.fftpack

SKIP          = 160                      # Numbers of samples to skip to start new frame
SAMPLES       = 400                      # Number of samples per frame
LOWER_FREQ    = 300.0                    # Lower frequency in Hz
UPPER_FREQ    = 8000.0                   # Upper frequency in Hz
FFT_LENGTH    = 512                      # Length of FFT
FFT_KEEP      = FFT_LENGTH // 2 + 1      # Number of FFT coefficients to keep
NFILTERS      = 26                       # Number of filterbanks for
Fs            = 16000                    # Sampling frequency in Hz
FRAME_LEN     = 0.025                    # Length of each frame in seconds
AUDIO_LEN     = 4                        # Length of audio sample in seconds
M             = 2                        # Window size of delta features
NCOEFFICIENTS = 13                       # Number of MFCC coefficients to keep

# Import data
raw = genfromtxt("s-data.csv", delimiter = ",")

# NOTE: ROWS is actually columns. Need to fix.
# Perform the DFT and create the power spectrum
ROWS = (Fs * AUDIO_LEN - SAMPLES) // SKIP + 1
power_spec = np.zeros([ROWS, FFT_LENGTH])
index = 0
for i in range(0, len(raw) - SAMPLES, SKIP):
    vector = []
    for j in range(i, i + SAMPLES):
        vector.append(raw[j])
    power_spec[index, :] = np.multiply((np.power(np.absolute(np.fft.fft(vector, n = FFT_LENGTH)), 2)), 0.0025)
    index = index + 1
np.savetxt("power_spec.csv", power_spec, delimiter = ",")

# Trim the power spectrum
trimmed_power_spec = np.zeros([ROWS, FFT_KEEP])
for i in range(0, ROWS):
    trimmed_power_spec[i, :] = power_spec[i, 0:FFT_KEEP]

# Find Mels of lower and upper frequencies
LOWER_MEL = 1125.0 * math.log(1.0 + (LOWER_FREQ / 700.0))
UPPER_MEL = 1125.0 * math.log(1.0 + (UPPER_FREQ / 700.0))

# Create the filterbanks in Mels
mel_banks = np.linspace(LOWER_MEL, UPPER_MEL, num = NFILTERS + 2)

# Convert the filterbanks back to frequency
freq_banks = [];
for i in range(0, len(mel_banks)):
    freq_banks.append(700.0 * (math.exp(mel_banks[i] / 1125.0) - 1.0))

bins = []
for i in range(0, len(freq_banks)):
    bins.append(math.floor((FFT_KEEP + 1.0) * freq_banks[i] / 8000))

# Make the Mel filterbank
filterbank = np.zeros([NFILTERS, FFT_KEEP])
for i in range(0, NFILTERS):
    # For each filter in the filterbank
    for j in range(int(bins[i]), int(bins[i + 1])):
        filterbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
    for j in range(int(bins[i + 1]), int(bins[i + 2])):
        filterbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])
np.savetxt("filterbank.csv", filterbank, delimiter = ",")

# x = np.zeros([NFILTERS, FFT_KEEP])
# for i in range(0, NFILTERS):
#     x[i, :] = np.linspace(LOWER_FREQ, UPPER_FREQ, FFT_KEEP)
# # plt.plot(x, filterbank, 'b.-')
# # plt.show()

# Collect coefficients/filterbank energies
energies = np.zeros([ROWS, NFILTERS])
for i in range(0, ROWS):
    # For each frame
    for j in range(0, NFILTERS):
        # For each filter in the filterbank
        energies[i, j] = np.dot(trimmed_power_spec[i, :], np.transpose(filterbank[j, :]))
np.savetxt("energies.csv", energies, delimiter = ",")

# Take the natural log and then the inverse Fourier transform of the energies
log_energies = np.log(energies)
np.savetxt("log_energies.csv", log_energies, delimiter = ",")

ifft_log_energies = scipy.fftpack.dct(log_energies)
np.savetxt("ifft_log_energies.csv", ifft_log_energies, delimiter = ",")

# Trim to keep only the first 13 energies
# Save the matrix in a 13 x T matrix
mfcc = np.transpose(ifft_log_energies[:, 0:NCOEFFICIENTS])
np.savetxt("mfcc.csv", mfcc, delimiter = ",")

# Find deltas
deltas = np.zeros([NCOEFFICIENTS, ROWS - 2 * M])
for i in range(0, NCOEFFICIENTS):
    # For each row in the MFCC matrix
    for j in range(0, ROWS - 2 * M):
        # Calculate summation to find delta coefficient
        d = 0
        for n in range(1, M + 1):
            d = d + ((n * (mfcc[i, j + 2 * n] - mfcc[i, j])) / (2 * (n ** 2)))
        deltas[i, j] = d
np.savetxt("deltas.csv", deltas, delimiter = ",")

# Trim the original MFCC matrix
cut_mfcc = np.zeros([NCOEFFICIENTS, ROWS - 2 * M])
for i in range(0, NCOEFFICIENTS):
    cut_mfcc[i, :] = mfcc[i, 0:ROWS - 2 * M]
np.savetxt("cut_mfcc.csv", cut_mfcc, delimiter = ",")

# Concatenate the original MFCC matrix with the delta matrix
# The MFCC matrix is above the delta matrix
final = np.zeros([NCOEFFICIENTS * 2, ROWS - 2 * M])
for i in range(0, NCOEFFICIENTS):
    final[i, :] = cut_mfcc[i, :]
    final[i + NCOEFFICIENTS, :] = deltas[i, :]
np.savetxt("final.csv", final, delimiter = ",")

# Create spectrograms
img = plt.imshow(final, interpolation='none', aspect = "auto")
plt.colorbar(img, orientation = "horizontal")
plt.show()
