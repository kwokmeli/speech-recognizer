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

AUDIOFILE = "/Users/mel/Desktop/PMP/E E 516/hw3/Phrases/What time is it/time10.csv"
PNGSAVELOCATION = "/Users/mel/Desktop/PMP/E E 516/hw3/p4/time5-plot.png"
SPECSAVELOCATION = "/Users/mel/Desktop/PMP/E E 516/hw3/p4/time5-spec.png"

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
print "ITL: %f" % ITL
print "ITU: %f" % ITU

# Find the zero crossing array
ZCsn = np.zeros([(SAMPLE_LENGTH * Fs - int(L)) / int(L), 1])
for i in range(0, (SAMPLE_LENGTH * Fs - int(L)) / int(L)):
    # For each row
    zcs = 0
    for j in range(0, int(L) - 1):
        # For each sample in a frame
        zcs = zcs + np.absolute(sgn(s[i, j + 1]) - sgn([i, j])) / 2
    ZCsn[i, :] = (1 / L) * zcs
np.savetxt("ZCsn.csv", ZCsn, delimiter = ",")

# Find the zero crossing array of the first 100 ms of silence
SILENCE_SAMPLES = int(Fs * 0.1)
ZCsn_silence = np.zeros([int(SILENCE_SAMPLES / L), 1])
for i in range(0, int(SILENCE_SAMPLES / L)):
    ZCsn_silence[i, :] = ZCsn[i, :]
np.savetxt("ZCsn_silence.csv", ZCsn_silence, delimiter = ",")

# Find the zero crossing threshold
IZC_mean = np.average(ZCsn_silence)
IZC_std = np.std(ZCsn_silence)
IZCT = min(IF, IZC_mean + 2 * IZC_std)
print "IZCT: %f" % IZCT

# Find frame N1, where speech begins
mglobal = 0
iglobal = None
N1 = None

def greaterThanOrEqualITL(m):
    while Esn[m, 0] < ITL:
        m = m + 1
    global mglobal
    global iglobal
    mglobal = m
    iglobal = m
    return m

def lessThanITL(i):
    global mglobal
    global iglobal
    global N1
    if Esn[i, 0] < ITL:
        mglobal = i + 1
        lessThanITL(greaterThanOrEqualITL(i + 1))
    else:
        if Esn[i, 0] >= ITL:
            N1 = i
            if iglobal == mglobal:
                N1 = N1 - 1
        else:
            iglobal = i + 1
            lessThanITL(iglobal)

lessThanITL(greaterThanOrEqualITL(mglobal))
print "N1: %d" % N1

# Find frame N2, where the speech ends
mglobal = len(Esn) - 1
iglobal = None
N2 = None

def N2greaterThanOrEqualITL(m):
    while Esn[m, 0] < ITL:
        m = m - 1
    global mglobal
    global iglobal
    mglobal = m
    iglobal = m
    return m

def N2lessThanITL(i):
    global mglobal
    global iglobal
    global N2
    if Esn[i, 0] < ITL:
        mglobal = i - 1
        N2lessThanITL(N2greaterThanOrEqualITL(i - 1))
    else:
        if Esn[i, 0] >= ITL:
            N2 = i
            if iglobal == mglobal:
                N2 = N2 + 1
        else:
            iglobal = i - 1
            N2lessThanITL(iglobal)

N2lessThanITL(N2greaterThanOrEqualITL(mglobal))
print "N2: %d" % N2

# Plot the signals
plt.figure(1)
plt.suptitle('Phrase: "What time is it"')
plt.subplot(211)
plt.plot(ZCsn)
plt.title("Plot of Zcs(n) vs. Frame Number")
plt.xlabel("Frame")
plt.ylabel("ZCs(n)")
plt.axvline(x = N1, color = "r", linestyle = "--")
plt.text(N1 + 4, 0.95 * max(ZCsn), "N1", color = "r")
plt.axvline(x = N2, color = "r", linestyle = "--")
plt.text(N2 + 4, 0.95 * max(ZCsn), "N2", color = "r")

plt.subplot(212)
plt.plot(Esn)
plt.title("Plot of Es(n) vs. Frame Number")
plt.xlabel("Frame")
plt.ylabel("Es(n)")
plt.axvline(x = N1, color = "r", linestyle = "--")
plt.text(N1 + 4, 0.95 * max(Esn), "N1", color = "r")
plt.axvline(x = N2, color = "r", linestyle = "--")
plt.text(N2 + 4, 0.95 * max(Esn), "N2", color = "r")
plt.savefig(PNGSAVELOCATION, bbox_inches = "tight")
plt.tight_layout()
# plt.show()

# Plot the spectrograms
plt.figure()
plt.title("Spectrogram of Es(n)")
Pxx, freqs, bins, im = plt.specgram(np.transpose(Esn[:,0]))
plt.savefig(SPECSAVELOCATION, bbox_inches = "tight")
# plt.show()
