import pyaudio
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from numpy import genfromtxt
import wave
import math
import scipy.fftpack

WINDOW = 0.01 # Window size in seconds
OVERLAP = 0.01 # Overlap size in seconds
Fs = 16000 # Sampling rate in Hz
SAMPLE_LENGTH = 4 # Length of audio sample in seconds
IF = 25

AUDIOFILE = "/Users/mel/Desktop/PMP/E E 516/hw3/Phrases/What time is it/time1.csv"
SAVELOCATION = "/Users/mel/Desktop/PMP/E E 516/hw3/p3/time1-matrix.csv"
FIGLOCATION = "/Users/mel/Desktop/PMP/E E 516/hw3/p3/time1-spec.png"

L = WINDOW * Fs

def sgn(x):
    if x >= 0:
        return 1
    else:
        return -1

# n is a 1 x M vector, where M is the number of samples
def Es(n):
    signal = 0
    for i in range(0, int(L)):
        signal = signal + np.absolute(n[i])
    return signal

# Import data
raw = genfromtxt(AUDIOFILE, delimiter = ",")

# Break imported data into frames
s = np.zeros([(SAMPLE_LENGTH * Fs - int(L)) / int(L), int(L)])
row_index = 0
for i in range(0, SAMPLE_LENGTH * Fs - int(L), int(L)):
    column_index = 0
    for j in range(i, int(i + L)):
        s[row_index, column_index] = raw[j]
        column_index = column_index + 1
    row_index = row_index + 1
np.savetxt("frames.csv", s, delimiter = ",")

# Create Es(n)
Esn = np.zeros([(SAMPLE_LENGTH * Fs - int(L)) / int(L), 1])
for i in range(0, (SAMPLE_LENGTH * Fs - int(L)) / int(L)):
    # For each row
    Esn[i, :] = Es(s[i, :])
np.savetxt("Esn.csv", Esn, delimiter = ",")

# Find the minimum and maximum of Es(n)
IMN = min(Esn)
IMX = max(Esn)

# Find ITL and ITU, the lower and upper thresholds of the energy signal
ITL = min(0.03 * (IMX - IMN) + IMN, 4 * IMN)
ITU = 5 * ITL

# Find the zero crossing array of the first 100 ms of silence
SILENCE_SAMPLES = int(Fs * 0.1)
ZCsn = np.zeros([int(SILENCE_SAMPLES / L), 1])
for i in range(0, int(SILENCE_SAMPLES / L)):
    # For the frames that are within the first 100 ms of silence
    zcs = 0
    for j in range(0, int(L) - 1):
        # For each sample in a frame
        zcs = zcs + np.absolute(sgn(s[i, j + 1]) - sgn([i, j])) / 2
    ZCsn[i, :] = (1 / L) * zcs
np.savetxt("ZCsn.csv", ZCsn, delimiter = ",")

# Find the zero crossing threshold
IZC_mean = np.average(ZCsn)
IZC_std = np.std(ZCsn)
IZCT = min(IF, IZC_mean + 2 * IZC_std)

# # Plot the signal
# plt.plot(Esn)
# plt.xlabel("Time")
# plt.ylabel("Es(n)")
# plt.show()
