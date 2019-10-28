# -*- coding: utf-8 -*-
# Oblig 3 - Toril Sunde Apelthun

from __future__ import division
import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
import scipy.io
import regression

# Endre denne tilbake til bare arbeidskrav3.mat
data = scipy.io.loadmat("/Users/torilapelthun/Documents/HVL/mat102/oblig3/oblig3/arbeidskrav3.mat")
x = np.array(data['x']).astype(float).reshape(11)
y = np.array(data['y']).astype(float).reshape(11)
omega = data["omega"][0][0].astype(float)

# 1a
# Linear regression
[a,b] = regression.linearRegression(x,y)
plt.figure(0)
plt.scatter(x,y)
xplot = np.linspace(-2,12)
yplot = np.dot(a,xplot)+b
plt.plot(xplot, yplot)
Sy2 = sum((y-np.mean(y))**2)
SSELin = sum((y-np.dot(a,x)-b)**2)
r2Linear = (Sy2-SSELin)/Sy2

# Quadratic regression
[a,b,c] = regression.quadraticRegression(x,y)
yplot = np.dot(a,np.power(xplot,2))+np.dot(b,xplot) + c
plt.plot(xplot,yplot)
SSEQuad = sum((y-(np.dot(a,np.power(x,2))+np.dot(b,x) + c))**2)
r2Quadratic = (Sy2-SSEQuad)/Sy2

# Cubic regression
[a,b,c,d] = regression.cubicRegression(x,y)
yplot =np.dot(a,np.power(xplot,3))+np.dot(b,np.power(xplot,2))+np.dot(c,xplot) + d
plt.plot(xplot,yplot)
SSECubic =sum((y-(np.dot(a,np.power(x,3))+np.dot(b,np.power(x,2))+np.dot(c,x) + d))**2)
r2Cubic = (Sy2-SSECubic)/Sy2
omega = 8

# 1b
print('Linear coefficient of determination: ', r2Linear) # 0.00322
print('Quadratic coefficient of determination: ', r2Quadratic) # 0.274
print('Cubic coefficient of determination: ', r2Cubic) # 0.686

# Determinasjonskoeffisienten skal være et tall mellom 0 og 1, og dess nærmere
# den er 1 dess bedre vil modellen stemme med punktene fra input-dataen.
# Her vil da den kubiske modellen være den modellen som er best egnet, etterfulgt
# av kvadartisk og lineær er minst egnet.
# Vi kan òg se dette på grafen som kommer opp når vi kjører koden.


# 1c - Sinusoid Regression
[a0,a1,b1] = regression.sinusoidRegression(x,y, omega)
plt.scatter(x,y)
xplot = np.linspace(-2,12)
yplot = a0+a1*np.cos(np.multiply(2*np.pi/omega,xplot))+b1*np.sin(np.multiply(2*np.pi/omega,xplot))
plt.plot(xplot,yplot)
Sy2 = sum((y-np.mean(y))**2)
SSESinus = sum((y-(a0+a1*np.cos(np.multiply(2*np.pi/omega,x))+b1*np.sin(np.multiply(2*np.pi/omega,x))))**2)
r2Sinus =(Sy2-SSESinus)/Sy2
print('Sinusoid coefficient of determination: ', r2Sinus) # 0.945

# Dette er defintivit den beste modellen for målingene våre.
# Den har en Determinasjonskoeffisient på 0.945, noe som er veldig nærme 1.
# Dette kan vi òg se når vi kjører koden og får opp modellen.

# 1d - X-verdi


plt.show()
