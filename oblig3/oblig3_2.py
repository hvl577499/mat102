# -*- coding: utf-8 -*-
from __future__ import division
import pca
import scipy.io
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import pylab
# import panda as pd

data = scipy.io.loadmat("/Users/torilapelthun/Documents/HVL/mat102/oblig3/arbeidskrav3.mat")
X1 = np.array(data['X1'])
X2 = np.array(data['X2'])

# 2a:
X1 = pca.meanCenter(X1) # Preprosessering av matrisen. Gjennomsnitt = 0 i hver søyle/kolonne.
X1 = pca.standardize(X1) # Preprossesering av matrisen. standardavvik = 1 i hver søyle/kolonne.
objNames1 = data['objNames1']
varNames1 = data['varNames1']
objNames2 = data['objNames2']
varNames2 = data['varNames2']

# 2b:
# Vi bruker PCA til å beskrive varians i datasettet vårt, og derfor er det
# viktig med standardisering. Ved standardisering får vi sammenliknbare
# resultater, uavhengig hvilken måleenhet eller benevning som var brukt på
# dataene i utgangspunktet.

# 2c
[T, P, E] = pca.pca(X1, a=2) # Setter a=2 fordi vi skal ha de to første prinsipalkomponentene

punktnavn = ['1:Milk+','2','3:Sugar','4','5a','5b','6','7:Cocoa+'] # Navn på punktene i score plot
pktnavn = ['%COCOA','%SUGAR','%MILK','COLOUR(L)','VISCOSITY/10','colour','cocoa-odour','smooth-txtr','milk-taste','sweet'] # Navn på punktene i loading plot

# 2c: Score plot
plt.figure(0)
plt.title('Score plot')
plt.scatter(T[:, 0], T[:, 1]) # Datapkt for de to første prinsipalkomponentene
for label, x, y in zip(objNames1, T[:, 0], T[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (5, -3),
        textcoords = 'offset points', ha = 'left')

# 2d:
# Ja, de to forsøkene med innstilling 5 er nær hverandre. Se vedlagt fil (2d.png) av plottingen.

# 2e: Hvor stor del av variasjonen er forklart?
varX1 = np.trace(np.dot(X1, X1.T))
print(varX1) # = 70%
varT = np.trace(np.dot(T, T.T))
print(varT) # 68.85%
# VarT/VarX1 = 0.98 => 98% av variasjonen er forklart

# 2f - Plotter loadings plot:
plt.figure(1)
plt.title('Loading plot')
plt.scatter(P[:, 0], P[:, 1])
for label, x, y in zip(varNames1, P[:, 0], P[:, 1]):
    plt.annotate(
        label,
        xy = (x, y), xytext = (5, -3),
        textcoords = 'offset points', ha = 'left')

# 2f - Plotter subplot
plt.figure(2)
plt.title('Biplot')
ax = pylab.subplot(111)
ax.scatter(T[:,0],T[:,1])
ax.scatter(P[:,0],P[:,1])

# Minst en kommentar til dataene basert på plottingen og analysen
#

plt.show()
