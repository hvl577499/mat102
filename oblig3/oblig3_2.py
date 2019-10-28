# -*- coding: utf-8 -*-
from __future__ import division
import pca
import scipy.io
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import pylab
import panda as pd

data = scipy.io.loadmat("/Users/torilapelthun/Documents/HVL/mat102/oblig3/arbeidskrav3.mat")
X1 = np.array(data['X1'])
X2 = np.array(data['X2'])

X1 = pca.meanCenter(X1)
X1 = pca.standardize(X1)
