from python_speech_features import mfcc
from python_speech_features import delta
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
from numpy import genfromtxt
import math
import scipy.fftpack

# Create spectrograms
LOCATION = "/Users/mel/Desktop/Speech-Recognizer/Samples/Odessa/MFCCs/od1.csv"
mfcc = genfromtxt(LOCATION, delimiter = ",")
img = plt.imshow(mfcc, interpolation='none', aspect = "auto")
plt.colorbar(img, orientation = "horizontal")
plt.show()
