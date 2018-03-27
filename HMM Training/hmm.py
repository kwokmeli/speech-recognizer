import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from numpy import genfromtxt
import wave
import math
import scipy.fftpack
import random
from functions import *
import time

Mw = 10      # Number of states in the HMM for a phrase w
N = 26      # Feature length
T = 399     # Width

PHRASE = ["Odessa"]
FILENAME = ["od"]

# nFILES = 10

nFILES = 1

# # Create MFCC data
# for phr in range(0, len(PHRASE)):
#     # For each phrase
#     for sample in range(0, nFILES):
#         # For each sample of the phrase
#         LOCATION = "/Users/mel/Desktop/Speech-Recognizer/Samples/" + PHRASE[phr] + "/Audio/" + FILENAME[phr] + str(sample + 1) + ".csv"
#         SAVELOCATION = "/Users/mel/Desktop/Speech-Recognizer/Samples/" + PHRASE[phr] + "/MFCCs/" + FILENAME[phr] + str(sample + 1) + ".csv"
#         m = MFCC(LOCATION, SAVELOCATION)
#     print("Finished MFCCs for " + PHRASE[phr])
#
# # Calculate the initial mean vector mu, which is the mean of all the MFCC vectors
# ugVector = np.zeros([N, 1])
# for phr in range(0, len(PHRASE)):
#     for sample in range(0, nFILES):
#         LOCATION = "/Users/mel/Desktop/Speech-Recognizer/Samples/" + PHRASE[phr] + "/MFCCs/" + FILENAME[phr] + str(sample + 1) + ".csv"
#         x = genfromtxt(LOCATION, delimiter = ",")
#         for t in range(0, T):
#             ugVector[:, 0] = ugVector[:, 0] + x[:, t]
# ugVector = ugVector * (1.0 / (T * nFILES * len(PHRASE)))
#
# # Calculate the initial covariance vector Cyt, which is the variance of all the MFCC vectors
# cytVector = np.zeros([N, 1])
# for phr in range(0, len(PHRASE)):
#     for sample in range(0, nFILES):
#         LOCATION = "/Users/mel/Desktop/Speech-Recognizer/Samples/" + PHRASE[phr] + "/MFCCs/" + FILENAME[phr] + str(sample + 1) + ".csv"
#         x = genfromtxt(LOCATION, delimiter = ",")
#         for t in range(0, T):
#             cytVector[:, 0] = cytVector[:, 0] + np.multiply(x[:, t] - ugVector[:, 0], x[:, t] - ugVector[:, 0])
# cytVector = cytVector * (1.0 / (T * nFILES * len(PHRASE)))
#
# # Create initial mean matrices for all the samples
# # Create the initial covariance matrices for all the samples
# for phr in range(0, len(PHRASE)):
#     for sample in range(0, nFILES):
#         uytSAVELOCATION = "/Users/mel/Desktop/Speech-Recognizer/Samples/" + PHRASE[phr] + "/uyt/" + FILENAME[phr] + str(sample + 1) + ".csv"
#         cytSAVELOCATION = "/Users/mel/Desktop/Speech-Recognizer/Samples/" + PHRASE[phr] + "/Cyt/" + FILENAME[phr] + str(sample + 1) + ".csv"
#         uyt = np.zeros([N, Mw])
#         Cyt = np.zeros([N, Mw])
#         for i in range(0, Mw):
#             for j in range(0, N):
#                 noise = np.random.randn()
#                 uyt[j, i] = ugVector[j, 0] + (noise * 1.0 / 8.0)
#                 Cyt[j, i] = cytVector[j, 0]
#         np.savetxt(uytSAVELOCATION, uyt, delimiter = ",")
#         np.savetxt(cytSAVELOCATION, Cyt, delimiter = ",")

# Initialize the transition probability matrix A so that it is upper triangular
# with random positive values between 0 and 1
A = np.zeros([Mw, Mw])
for i in range(0, Mw):
    # For each row
    if i != Mw - 1:
        A[i, i] = 0.5
        A[i, i + 1] = 0.5
    else:
        A[i, i] = 1

# Create the alpha matrices for all the audio samples
start = time.time()
for phr in range(0, len(PHRASE)):
    for sample in range(0, nFILES):
        LOCATION = "/Users/mel/Desktop/PMP/E E 516/hw5/Phrases/" + PHRASE[phr] + "/MFCCs/" + FILENAME[phr] + str(sample + 1) + ".csv"
        x = genfromtxt(LOCATION, delimiter = ",")
        alpha = np.zeros([Mw, T])

        # Initialize the first column of the alpha matrix
        alpha[0, 0] = 1

        # Fill in the rest of the alpha matrix
        for t in range(1, T):
            csum = 0
            for j in range(0, Mw):
                psum = 0
                for i in range(0, Mw):
                    if i == j or j == i + 1:
                        psum = np.add(psum, alpha[i, t - 1] * A[i, j])
                        # psum = psum + alpha[i, t - 1] * A[i, j]
                csum = np.add(csum, psum)
                alpha[j, t] = psum * p_xt_yt(PHRASE[phr], FILENAME[phr], sample, t, j)
                # csum = csum + psum
                # alpha[j, t] = psum * p_xt_yt(PHRASE[phr], FILENAME[phr], sample, t, j) / csum
            alpha[:, t] = np.divide(alpha[:, t], csum)

        print "Finished alpha matrix " + FILENAME[phr] + str(sample + 1)

        # Save the alpha matrix
        SAVELOCATION = "/Users/mel/Desktop/PMP/E E 516/hw5/Phrases/" + PHRASE[phr] + "/Alpha/" + FILENAME[phr] + "-alpha" + str(sample + 1) + ".csv"
        np.savetxt(SAVELOCATION, alpha, delimiter = ",")
end = time.time()
print "Alpha time: %f" % (end - start)
