#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 08:31:11 2020

@author: evelyn
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname='points',delimiter='\t')
x = data[:,0]
x_err = data[:,1]
y = data[:,2]
y_err = data[:,3]
x_mean = np.mean(x)


y_mean = np.mean(y)

xy = x*y
x2 = x*x
x_sum = np.sum(x)
y_sum = np.sum(y)
samples = data.shape[0]

slope    = (np.mean(xy) - np.mean(x)*np.mean(y))\
	 / (np.mean(x2) - np.mean(x)**2)

intercept = (np.mean(x2) * np.mean(y) - np.mean(x) * np.mean(xy))\
	 / (np.mean(x2) - np.mean(x)**2)

fit = np.polyfit(x,y,1,full=True)
a = fit[0]
b = fit[1]

y_fit = slope*x+intercept
fig, ax = plt.subplots()
ls = 'None'
ax.errorbar(x,y,yerr=y_err,uplims=True, lolims=True,\
			 color='black',marker='.',markersize=8,linestyle=ls,\
 				 label='Experimental Data')
plt.plot(x,y_fit,color='red',label='Linear Fit')

plt.legend()

A1=x-np.mean(x)
A2=y-np.mean(y)

R2 = ( (np.sum(A1*A2))**2 )/ \
 	( np.sum((A1**2))*np.sum((A2**2)) )
print("R2 = ",R2)
print(fit)
print("Manual      :",(slope,intercept))


# =============================================================================
# Ajuste a un polinomio Pn(x)
# =============================================================================
# Generamos algunos puntos para probar el ajuste

np.random.seed(1) # fijando la semilla del generador para asegurar reproducibilidad
N=13   # orden del polinomio a ajustar

# Dominio de la funcion
x = np.array([i*np.pi/180 for i in range(-360,360,10)])

# Funcion con fluctuaciones aleatorias
y = np.sin(x) + np.cos(2*x) + np.random.normal(0,.15,len(x))

# Funcion real a la que se le provocan dichas fluctuaciones
y_real = np.sin(x) + np.cos(2*x)

fig,axs=plt.subplots(1)

# =============================================================================
# Construir la matriz de los coeficientes
# la ecuacion n-esima tiene la forma:
# Sum(aj)_jSum(xi^(n+j))_i
# por tanto el coeficiente i,j de la matriz tiene la forma Sum(xi^(n+j))_i
# =============================================================================
orden = N+1;
A=np.ndarray((orden,orden))   # A es la matriz de coeficientes
b=np.ndarray((orden,1))       # b es el vector adicional de la matriz extendida (A|b)
for i in range(orden): # este loop va por las ecuaciones
	for j in range(orden): # este loop va por los coeficientes
		A[i,j] = np.sum(x**(i+j))
for i in range(0,orden,1):
	b[i] = np.sum(y*(x**i))
	
# Resolver el sistema 
sol = np.linalg.inv(A).dot(b)
fit = np.polyfit(x,y,N) # Determinamos los coeficientes con un metodo
                                  # validado para comparar

# Este metodo construye el vector de la imagen usando los coeficientes 
# determinados por el ajuste
def GetFittedY(sol_vector):
	y_fit = np.ndarray((len(x)))
	for i in range(0,len(x),1):
		Px=0
		for c in range(len(sol_vector)):
			Px += (x[i]**c) * sol_vector[c]
		y_fit[i] = Px
	return y_fit

y_fitted = GetFittedY(sol)

# Visualizar los resultados
axs.scatter(x,y, label='Data')
#axs.plot(x,y_fitted,color='red', label='Polinomic fit')
axs.plot(x,y_real,color='black', label='Funcion real')
plt.legend()


# =============================================================================
# Create an animation varying the polinomia order
# =============================================================================
def yfitted(power):
	power = power%20
	fit = np.polyfit(x,y,power)
	return GetFittedY(fit[::-1])
	
import matplotlib.animation as animation
line, = axs.plot(x, yfitted(1),label='Polinomic fit',color='red')
plt.legend()
def animate(i):
    line.set_ydata(yfitted(i))  # update the data.
    return line,

ani = animation.FuncAnimation(
    fig, animate, interval=400, blit=True, save_count=50)
