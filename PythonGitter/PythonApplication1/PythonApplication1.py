#!/usr/bin/env python
# -*- coding: utf-8; -*-

# 2017/2018 
# Authors: Bernd Lienau, Simon Jung, Alexander Franke
# published under GNU General Public License v3.0
# see LICENSE file for further details

import sys
import time

import math

import numpy as np
from numpy import sin as sin
from numpy import cos as cos
from numpy import tan as tan
from numpy import arcsin as arcsin
from numpy import arccos as arccos
from numpy import arctan as arctan
from numpy import exp as exp
from numpy.fft import fft as fft
from numpy.fft import fft2 as fft2
from numpy import pi as pi

#plots
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import matplotlib.pyplot as plt

 
import pandas as pd
import scipy
from scipy import integrate as integrate
import random

import argparse

#Canvas
from tkinter import *

####### TODOS ######
'''
	- Canvas result update instead of redraw

moeglich:
	- Uberlagerung von verschiendenen Wellenlaengen
	- mehrere Gitter
	- Einfallswinkel von 90 deg verschieden


'''

def main():
	parser = argparse.ArgumentParser(description='This is a python3 module simulating a light pulse with given parameters propagating through different optical components suchas Einzelspalt, Doppelspalt, Gitter mit und ohne Fehlstellen oder Defekten.')
	parser.add_argument('--nx', dest='nx', help='Die Anzahl der Spalte in x-Richtung. Ganzzahlige Zahl zwischen 0 und Unendlich. 0 steht hierbei fuer einen Spalt mit unendlicher Ausdehnung.',default=1)
	parser.add_argument('--ny', dest='ny', help='Die Anzahl der Spalte in y-Richtung. Ganzzahlige Zahl zwischen 0 und Unendlich. 0 steht hierbei fuer einen Spalt mit unendlicher Ausdehnung.',default=1)
	parser.add_argument('--ax', dest='ax', help='Spaltbreite in um',default=3)
	parser.add_argument('--ay', dest='ay', help='Spalthoehe in um',default=3)
	parser.add_argument('--dx', dest='dx', help='Spaltabstand in horizontaler Richtung in um',default=5)
	parser.add_argument('--dy', dest='dy', help='Spaltabstand in vertikaler Richtung in um',default=5)
	parser.add_argument('--errortype', dest='errortype', help='Gitterfehlertyp. 0 fuer keinen Fehler. 1 fuer zufaellige, kleine Verschiebung jedes Spaltes, 2 fuer 10% Chance fuer jedes Loch, dass es nicht existiert (Fehlstellen)',default=0)
	parser.add_argument('--wl', dest='wl',help='Wellenlänge in nm',default=780 )
	parser.add_argument('--zs', dest='zs', help='Schirmabstand in cm',default=350)
	parser.add_argument('--calctype', dest='calctype',help='Waehle aus default,canvas)',default='default')
	
	args = parser.parse_args()

	print('------------------------------------------------------------------------------')
	print('Kunst der Computer-basierten Modellierung und Simulation experimenteller Daten')
	print('------------------------------------------------------------------------------')
	print('')
	print('                        Projekt Lichtpuls Simulation                          ')
	print('')
	print('              von Bernd Lienau, Simon Jung, Alexander Franke                  ')
	print('')
	print('------------------------------------------------------------------------------')
	print('')
	print('Es wurden folgende Parameter eingegeben oder angenommen'                       )
	#print('   Dimension                                  :  ' + str(args.dimension))
	#print('   Falls Dimension = 1, berechnen wir für     :')
	print('   Anzahl der Spalte:')
	print('      nx=' + str(args.nx) + ' Spalte in horizontaler Richtung(x)'              )
	print('      ny=' + str(args.ny) + ' Spalte in vertikaler Richtung(y)'                )
	print('   Spaltbreite in um          :'                                               )
	print('      horizontal(x):   ax=' + str(args.ax)                                     )
	print('      vertikal(y):     ay=' + str(args.ay)                                     )
	print('   Spaltabstand in um          :')
	print('      horizontal(x):   dx=' + str(args.dx)                                     )
	print('      vertikal(y):     dy=' + str(args.dy)                                     )
	print('   Gitterfehler:          ' + str(args.errortype)                              )
	print("   Wellenlänge in  nm     " + str(args.wl)                                     )
	print("   Schirmabstand in cm    " + str(args.zs)                                     )
	print('                                                                              ')
	print('------------------------------------------------------------------------------')

	#__________________________________________________________________
	# Variablen auf SI Einheiten bringen. 
	nx = int(args.nx)
	ny = int(args.ny)
	ax = int(args.ax)  * 1e-6
	ay = int(args.ay)  * 1e-6
	dx = int(args.dx)  * 1e-6
	dy = int(args.dy)  * 1e-6
	errortype = int(args.errortype)
	wl = int(args.wl) * 1e-9
	zs = int(args.zs) * 1e-2
	calctype = str(args.calctype)
	
	
	matplotlib.rcParams.update({'font.size': 30}) ## change font size

	
	if nx==0 and ny==0:
		print('Ohne Gitter gibt es keine Berechnung...')
		sys.exit(0)


	if calctype == 'canvas': 
		# Leinwand aehnlich zu mspaint. Beugungsmuster eines beliebigen Objekts
		Main_Canvas(wl,zs)
	elif calctype == 'default':
		Main_Default(nx,ny,ax,ay,dx,dy,errortype,error_matrix(nx,ny,errortype),wl,zs)
		
	#__________________________________________________________________
	# Ende der main()


####__________________________________________________________________
#### Hilfsvariablen/funktionen. 
####__________________________________________________________________

### kgV2(int a, int b): 
### returns: KgV der beiden ints a und b zurück
def kgV2(a, b):
	return (a * b) // math.gcd(a, b)

### kgV_arr([int] numbers): 
### returns: kgV einer beliebigen Liste von Ints zurück
def kgV_arr(numbers):
	kgV = numbers[0]
	for i in range(1,len(numbers)):
		kgV=kgV2(kgV,numbers[i])
	return kgV

### k(int wl)
### returns Wellenzahl bei gegebener Wellenlaenge wl
def k(wl):
	return 2 * np.pi / wl 

### w(int wl)
### returns Kreisfrequenz Omega bei gegebener Wellenlaenge wl
def w(wl):
	return c() * k(wl)

### c()
### returns float Lichtgeschwindigkeit
def c():
	return float(299792458) # m/s

### sinc(float)
### returns sinc function: sin(x)/x
def sinc(x):
	return sin(x)/x

### i()
### returns the imaginary unit.
def i():
	return complex(0,1)

### abstandZweiterPkte(int x0, int y0, int x, int y)
### returns: Abstand des Punktes (x,y) zum Punkt (x0,y0) auf einer 2D Ebene
def abstandZweierPkte(x0,y0,x,y):
	return round(math.sqrt((x-x0)**2 + (y-y0)**2))

### complex_int(lambda y: func, int a, int b):
### returns: complex integral of func from a to b
def complex_int(func, a, b, **kwargs):
	def real_func(x):
		return scipy.real(func(x))
	def imag_func(x):
		return scipy.imag(func(x))
	real_integral = integrate.quad(real_func, a, b, **kwargs)
	imag_integral = integrate.quad(imag_func, a, b, **kwargs)
	return (real_integral[0] + i()*imag_integral[0])

### error_matrix(int nx, int ny):
### returns if type 1: Transformiert ein fehlerfreies Gitter definiert durch die Anzahl der Spalte in x- und y-Richtung nx und ny in ein Gitter mit leicht verschobenen Loechern
### returns if type 2: Transformiert ein fehlerfreies Gitter definiert durch die Anzahl der Spalte in x- und y-Richtung in eines mit Fehlstellen

def error_matrix(nx,ny,errortype):
	if (errortype ==2):
		error_matrix = []
		for i in range(ny):
			error_row=[]
			for j in range(nx):
				if(random.randint(0, 100) > 10):
					error_row.append([[1,1,1,1]])
				else:
					error_row.append([[0,0,0,0]])
			error_matrix.append(error_row)
		return np.array(error_matrix)
	else: #error type 1
		error_matrix = []
		for i in range(ny):
			error_row=[]
			for j in range(nx):
				error_row.append([[random.uniform(-0.2,0.2),random.uniform(0.9,1.1)],[random.uniform(-0.2,0.2),random.uniform(0.9,1.1)]])
			error_matrix.append(error_row)
		return np.array(error_matrix)


### formatSecToMillisec(float time)
### returns: converts entered sec time to ms, rounds and adds Einheit
def formatSecToMillisec(time):
	return str(round(time*1000)) + ' ms'

### class nlcmap()
### zur Darstellung der Intensität verwenden wir einen nicht-linearen, diskreten Farbverlauf nach 
### https://stackoverflow.com/questions/33873397/nonlinear-colormap-with-matplotlib
class nlcmap(LinearSegmentedColormap):
	name = 'nlcmap'
	def __init__(self, cmap, levels):
		self.cmap = cmap
		self.monochrome = self.cmap.monochrome
		self.levels = np.asarray(levels, dtype='float64')
		self._x = self.levels/ self.levels.max()
		self.levmax = self.levels.max()
		self.levmin = self.levels.min()
		self._y = np.linspace(self.levmin, self.levmax, len(self.levels))
	def __call__(self, xi, alpha=1.0, **kw):
		yi = np.interp(xi, self._x, self._y)
		return self.cmap(yi/self.levmax, alpha)

####__________________________________________________________________
#### Berechnungsfunktionen mittels Fouriertransformation 
####__________________________________________________________________ 
def fourierNspaltPeriodisch(xArray,yArray,nx,ny,ax,ay,dx,dy,wl,zs):  
	#Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
	#beim Integrieren bzw bei der fft schief geht.
	subArrayX= []
	subArrayY= []
	
	for x in xArray:
		if nx==0:
			subArrayX.append(1)
		else:
			subArrayX.append(float(fourierNspaltPeriodischIntegrate(x,nx,ax,dx,wl,zs)))
	for y in yArray:
		if ny==0:
			subArrayY.append(1)
		else:
			subArrayY.append(float(fourierNspaltPeriodischIntegrate(y,ny,ay,dy,wl,zs)))
		
	XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
	Ztmp=XX*YY

	return Ztmp

	
def fourierNspaltAnyFunction(xArray,yArray,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs): ##gibt 1D richtiges Ergebnis
	## bietet die Möglichkeit in 'fourierNspaltIntegrateWithWholeTransmissionFunction(x,nx,ax,dx,wl,zs)' eine
	## beliebige Funktion für das Gitter einzusetzen
	
	#Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
	#beim Integrieren bzw bei der fft schief geht.
	subArrayX= []
	subArrayY= []
	
	## Teile xArray an Mittelpunkten zwischen den Spalten in Teile von
	##     [:-(nx-2)/2*dx][-(nx-2)/2*dx:(i-(nx-2)/2*dx)][...] i in range(nx-1)
	## Teile yArray an Mittelpunkten zwischen den Spalten in Teile von
	##     [:-(ny-2)/2*dy][-(ny-2)/2*dy:(j-(ny-2)/2*dy)][...] j in range(ny-1)
	## Erhalte somit nx*ny Teilstücke des Gitters, in denen sich jeweils ein Spalt befindet
	## Integriere für jeden einzelnen Spalt separat, fülle die restlichen Gebiete des Ergebnisses mit 1
	## multipliziere die einzelnen Spaltfouriertransformierten um das Gesamtergebnis zu erhalten
	Ztmp=[]
	
	for i in range(nx):
		for j in range(ny):
			for x in xArray:
				if nx==0:
					subArrayX.append(1)
				elif x > ((i-1-(nx-2)/2)*dx) and x <= ((i-(nx-2)/2)*dx):
					subArrayX.append(float(fourierNspaltIntegrateAnyFunction(x,nx,ax,dx,errortype,error_matrix[j,:,0],wl,zs)))
				else:
					subArrayX.append(0)
			
			for y in yArray:
				if ny==0:
					subArrayY.append(1)
				elif y > ((j-1-(ny-2)/2)*dy) and y <= ((j-(ny-2)/2)*dy):
					subArrayY.append(float(fourierNspaltIntegrateAnyFunction(y,ny,ay,dy,errortype,error_matrix[:,i,1],wl,zs)))
				else:
					subArrayY.append(0)
					
			XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
			Ztmp.append(XX*YY)
			subArrayX= []
			subArrayY= []
	
	Ztotal=Ztmp[0]
	for k in range(1,len(Ztmp)):
		Ztotal+=Ztmp[k]
	return Ztotal

def fftCanvas2D_XYZ(imagearray,wl,zs): ## 1 pixel = 0.1 um
	## 2D Berechnung
	d = 1e-6
	n = 1
	a = 1e-6
	N = 800 ## Datenpunkte im ganzen Array, mit Anfang- und Endpunkt, daher +1
	x_Spalt = np.array(np.linspace(-N/2*1e-7,N/2*1e-7,N))   ## wähle großen Bereich für die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist
	y_Spalt = np.array(np.linspace(-N/2*1e-7,N/2*1e-7,N))   ## wähle großen Bereich für die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist
	
	z2D = np.hstack((np.zeros(shape=(imagearray.shape[0], int((N-imagearray.shape[1])/2))), imagearray,
					 np.zeros(shape=(imagearray.shape[0], int((N-imagearray.shape[1])/2)))))
	z2D = np.vstack((np.zeros(shape=(int((N-z2D.shape[0])/2),int(z2D.shape[1]))), z2D,
					 np.zeros(shape=(int((N-z2D.shape[0])/2),int(z2D.shape[1])))))
	
	deltax = (x_Spalt[1]-x_Spalt[0]) #Sampling-Rate ist für x- und y-Richtung gleich
	fa = 1.0/deltax #Nyquist-Frequenz
	Xf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs  #Datenpunkte der fft als k-Vektoren im np.linspace(..)
	# zurückgerechnet in x-/y-Positionen auf dem Schirm via Gl. LS(k) = integral(transmission(x)*exp(-2*pi*i*k*x)dx)
	# hierbei ist k die Wellenzahl und somit gibt LS(k)/k0=LS(k)*wl=sin(alphax) den Winkel zur Stahlachse an,
	# unter dem der gebeugte Strahl probagiert. Mit Hilfe des tan(alphax) und der Schirmentfernung zs findet sich
	# so durch tan(alphax)*wl=tan(arcsin(LS(k)*wl))*zs die x-Koordinate auf dem Schirm, zu der der k-Vektor der fft gehört.
	# So wird Xf berechnet, welches jedem Intensitätswert aus der fft den richtigen Punkt auf dem Schirm zuordnet
	Yf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs

	index_low =  np.argmax(Xf>-5.0) #Beschränke den Plot auf -5m bis +5m auf dem Screen
	index_high = np.argmax(Xf>5.0)
	if index_high==0:
		index_high = len(Xf)
	X_Schirm, Y_Schirm = np.meshgrid(Xf,Yf)#[index_low:index_high],Yf[index_low:index_high])
	
	z2Df = fftshift(np.square(np.abs(fft2(z2D))*4/N/N)) #[index_low:index_high,index_low:index_high]
	return X_Schirm, Y_Schirm, z2Df


def fftNspalt2D_XYZ(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs): ##Ergebnisform stimmt, Skalierung noch nicht
		
	
	if nx==0:  ## then only integrate in y-direction
		Schirm, z1Dfy = fftNspalt1D_XZ(ny,ay,dy,errortype,error_matrix[:,0,1],wl,zs) ## error_matrix[:,0,1] is the array of the error-values in y-direction
		z1Df_X, z1Df_Y = np.meshgrid(np.ones(len(z1Dfy)),z1Dfy)
		z2Df = z1Df_X * z1Df_Y
		X_Schirm, Y_Schirm = np.meshgrid(Schirm,Schirm)
	elif ny==0: ## then only integrate in x-direction
		Schirm, z1Dfx = fftNspalt1D_XZ(nx,ax,dx,errortype,error_matrix[0,:,0],wl,zs) ## error_matrix[0,:,0] is the array of the error-values in x-direction
		z1Df_X, z1Df_Y = np.meshgrid(z1Dfx,np.ones(len(z1Dfx)))
		z2Df = z1Df_X * z1Df_Y
		X_Schirm, Y_Schirm = np.meshgrid(Schirm,Schirm)
	else:
		## 2D Berechnung
		d = max(dx,dy)
		n = max(nx,ny)
		a = max(ax,ay)
		datapoints = kgV_arr([int(d*1e6*20*n),int(a*1e6)])  ## minimale Anzahl an Datenpunkten, damit an jedem Spaltrand ein Punkt liegt
		while(datapoints*wl/4/n/d/10<0.82*2):               ## erhöhe Datapoints, damit mindestens die Raumfrequenzen berechnet werden, die auf dem Schirm abgebildet werden
			datapoints*=2                                   ## 0.82*2 für Plot bis +-5m
		N = datapoints+1                                    ## Datenpunkte im ganzen Array, mit Anfang- und Endpunkt, daher +1
		x_Spalt = np.array(np.linspace(-n*d*10,n*d*10,N))   ## wähle großen Bereich für die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist
		y_Spalt = np.array(np.linspace(-n*d*10,n*d*10,N))   ## wähle großen Bereich für die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist

		deltax = (x_Spalt[1]-x_Spalt[0]) #Sampling-Rate ist für x- und y-Richtung gleich
		fa = 1.0/deltax #Nyquist-Frequenz
		Xf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs  #Datenpunkte der fft als k-Vektoren im np.linspace(..)
		# zurückgerechnet in x-/y-Positionen auf dem Schirm via Gl. LS(k) = integral(transmission(x)*exp(-2*pi*i*k*x)dx)
		# hierbei ist k die Wellenzahl und somit gibt LS(k)/k0=LS(k)*wl=sin(alphax) den Winkel zur Stahlachse an,
		# unter dem der gebeugte Strahl probagiert. Mit Hilfe des tan(alphax) und der Schirmentfernung zs findet sich
		# so durch tan(alphax)*wl=tan(arcsin(LS(k)*wl))*zs die x-Koordinate auf dem Schirm, zu der der k-Vektor der fft gehört.
		# So wird Xf berechnet, welches jedem Intensitätswert aus der fft den richtigen Punkt auf dem Schirm zuordnet
		Yf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs

		index_low =  np.argmax(Xf>-5.0) #Beschränke den Plot auf -5m bis +5m auf dem Screen
		index_high = np.argmax(Xf>5.0)
		if index_high==0:
			index_high = len(Xf)
		X_Schirm, Y_Schirm = np.meshgrid(Xf[index_low:index_high],Yf[index_low:index_high])
		
		z2D = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,errortype,error_matrix)
		z2Df = fftshift(np.square(np.abs(fft2(z2D))*4/N/N))[index_low:index_high,index_low:index_high]
	return X_Schirm, Y_Schirm, z2Df

def fftNspalt1D_XZ(nx,ax,dx,errortype,error_array,wl,zs):
	datapoints = kgV_arr([int(dx*1e6*20*nx),int(ax*1e6)]) ## minimale Anzahl an Datenpunkten, damit an jedem Spaltrand ein Punkt liegt
	while(datapoints*wl/4/nx/dx/10<0.82*2):                 ## erhöhe Datapoints, damit mindestens die Raumfrequenzen berechnet werden, die auf dem Schirm abgebildet werden
		datapoints*=2                                     ## 0.82 für Plot bis +-5m
	N = datapoints+1                                      ## Datenpunkte im ganzen Array, mit Anfang- und Endpunkt, daher +1
	
	x_Spalt = np.array(np.linspace(-dx*nx*10,dx*nx*10,N)) ## wähle großen Bereich für die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist
	
	deltax = (x_Spalt[1]-x_Spalt[0])
	fa = 1.0/deltax #Nyquist-Frequenz
	Xf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs  #Datenpunkte der fft als k-Vektoren im np.linspace(..)
	# zurückgerechnet in x-/y-Positionen auf dem Schirm via Gl. LS(k) = integral(transmission(x)*exp(-2*pi*i*k*x)dx)
	# hierbei ist k die Wellenzahl und somit gibt LS(k)/k0=LS(k)*wl=sin(alphax) den Winkel zur Stahlachse an,
	# unter dem der gebeugte Strahl probagiert. Mit Hilfe des tan(alphax) und der Schirmentfernung zs findet sich
	# so durch tan(alphax)*wl=tan(arcsin(LS(k)*wl))*zs die x-Koordinate auf dem Schirm, zu der der k-Vektor der fft gehört.
	# So wird Xf berechnet, welches jedem Intensitätswert aus der fft den richtigen Punkt auf dem Schirm zuordnet
	
	index_low = np.argmax(Xf>-5.0) #Beschränke den Plot auf -5m bis +5m auf dem Screen
	index_high = np.argmax(Xf>5.0)
	if index_high==0:
		index_high=len(Xf)
	X_Schirm = Xf[index_low:index_high]
	
	## 1D Berechnung
	Transmission_X = []
	for x in x_Spalt:
		Transmission_X.append(Transmission_n_Spalte(x,nx,ax,dx,errortype,error_array))
	z1Df = fftshift(np.square(np.abs(fft(Transmission_X))*2/N))
		
	return X_Schirm, z1Df[index_low:index_high]
	
def fourierNspaltIntegrateAnyFunction(x,n,a,d,errortype,error_array,wl,zs):
	# Fouriertransformierte von Transmission_Gitter
	
	## bietet die Möglichkeit eine beliebige Funktion für das Gitter in 'Transmission_n_Spalte(y,n,a,d)' einzusetzen
	
	u = k(wl)*sin(arctan(x/zs))
	#lambda x sagt python nur dass das die Variable ist und nach der integriert werden muss
	f = lambda y:  Transmission_n_Spalte(y,n,a,d,errortype,error_array)*exp(-i()*u*y) 

	integral = complex_int(f,-(n-1)*d/2-a,(n-1)*d/2+a)
	#scipy.real koennte man weg lassen, da korrekterweise der imaginaer Teil immer null ist. Aber damit
	#matplot keine Warnung ausgibt, schmeissen wir den img Teil hier weg.
	integral =  scipy.real(np.square(np.multiply(n,integral)))
	return integral


def fourierNspaltPeriodischIntegrate(x,n,a,d,wl,zs):
	# Fouriertransformierte von Transmission_Einzelspalt
	# Siehe Glg 29 im Theory doc.pdf
	# https://en.wikipedia.org/wiki/Dirac_delta_function#Translation folgend
	# ist f(t) gefaltet mit dirac(t-T) ist gleich f(t-T)
	# außerdem gilt distributivität (a+b) (*) c = a(*)c + b(*)c
	# für den Doppelspalt bzw. n-Spalt haben wir also
	u = k(wl)*sin(arctan(x/zs))
	#lambda x sagt python nur dass das die Variable ist und nach der integriert werden muss
	f = lambda y:  Transmission_Einzelspalt(y,a) *exp(-i()*u*y) 
	r = 0
	#Fuehre einen Multiplikationsfaktor ein. Dieser Faktor entspricht dem aus Glg 34 ff.
	#Fuer jeden Spalt finden wir den Mittelpunkt und addieren entsprechend die 
	#Fouriertransformation dieser Dirac funktion. Die Breite a dieser ganzen Spalte ist durch
	#die Funktion f mit der Transmission eines Spaltes festgelegt.
	#Hier ist also noch eine Verbesserung notwendig, die uns ermoeglicht unterschiedlich breite
	#Spalte einzubauen.

	mittelpunkteDerLoecher = Transmission_Mittelpunkte(n,d)
	for pkt in mittelpunkteDerLoecher:
		r = r + (exp(i()*u*pkt))

	if(n==1):
		r = 1
	integral = complex_int(f,-a,a)
	#scipy.real koennte man weg lassen, da korrekterweise der imaginaer Teil immer null ist. Aber damit
	#matplot keine Warnung ausgibt, schmeissen wir den img Teil hier weg.
	integral =  scipy.real(np.square(n * np.multiply(integral,r)))
	return integral

####__________________________________________________________________
#### Transmissionsfunktion verschiedener Objekte
####__________________________________________________________________

def Transmission_Einzelspalt(x,a):
	if math.fabs(x) <= a/2:
		return 1
	else:
		return 0

def Transmission_Lochblende(rho,R):
	#einzelnes Loch mit Radius R
	#Verwende Polarkoordinaten rho,theta 
	if (rho <= R):
		return 1
	else: 
		return 0
	
def Transmission_Mittelpunkte(n,d):
	mittelpunkte = []
	i = 1
	if (n%2) == 1:
		mittelpunkte.append(0)

	while i<=n/2:
		if (n % 2) == 0:
			mittelpunkte.append(d*(2*i-1)/2)
			mittelpunkte.append(-d*(2*i-1)/2)
		else:
			mittelpunkte.append((i)*d)
			mittelpunkte.append(-(i)*d)

		i =i+1

	return mittelpunkte

def Transmission_n_Spalte(x,n,a,d,errortype,error_array):
	gesamttransmission = 0.
	
	if errortype == 0:  ##kein Error
		for i in range(n):
			gesamttransmission += Transmission_Einzelspalt(x-d*(i-n/2+0.5),a)
	
	elif errortype == 1:
		for i in range(n):
			gesamttransmission += Transmission_Einzelspalt(x+error_array[i,0]*a-d*(i-n/2+0.5),a*error_array[i,1])
			
	elif errortype == 2:
		#there is a 10% chance that an hole is missing
		for i in range(n):
			gesamttransmission += Transmission_Einzelspalt(x-d*(i-n/2+0.5),a) * int(error_array[i][0])

	return gesamttransmission

def Transmission_Gitter(xArray,yArray,nx,ny,ax,ay,dx,dy,errortype,error_matrix):
	# Returns the transmission for a periodic grid as a matrix with 0/1
	# to plot it with the contourplot-fct or use it for the fft2 algorithm
	# error is the error of the grid which is given to the Transmission-fct
	subArrayX=[]
	subArrayY=[]
	
	## Teile xArray an Mittelpunkten zwischen den Spalten in Teile von
	##     [:-(nx-2)/2*dx][-(nx-2)/2*dx:(i-(nx-2)/2*dx)][...] i in range(nx-1)
	## Teile yArray an Mittelpunkten zwischen den Spalten in Teile von
	##     [:-(ny-2)/2*dy][-(ny-2)/2*dy:(j-(ny-2)/2*dy)][...] j in range(ny-1)
	## Erhalte somit nx*ny Teilstücke des Gitters, in denen sich jeweils ein Spalt befindet
	## Integriere für jeden einzelnen Spalt separat, fülle die restlichen Gebiete des Ergebnisses mit 1
	## multipliziere die einzelnen Spaltfouriertransformierten um das Gesamtergebnis zu erhalten
	Ztmp=[]
	
	for i in range(max(nx,1)):
		for j in range(max(ny,1)):
			for x in xArray:
				if nx==0:
					subArrayX.append(1)
				elif x > ((i-1-(nx-2)/2)*dx) and x <= ((i-(nx-2)/2)*dx):
					subArrayX.append(Transmission_n_Spalte(x,nx,ax,dx,errortype,error_matrix[j,:,0]))
				else:
					subArrayX.append(0)
			for y in yArray:
				if ny==0:
					subArrayY.append(1)
				elif y > ((j-1-(ny-2)/2)*dy) and y <= ((j-(ny-2)/2)*dy):
					subArrayY.append(Transmission_n_Spalte(y,ny,ay,dy,errortype,error_matrix[:,i,0]))
				else:
					subArrayY.append(0)
					
			XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
			Ztmp.append(XX*YY)
			subArrayX= []
			subArrayY= []
	
	Ztotal=Ztmp[0]
	for k in range(1,len(Ztmp)):
		Ztotal+=Ztmp[k]
	
	return np.array(Ztotal)
			
####__________________________________________________________________
#### Intensitätsverteilungen für verschiedene Objekte. Ich weiß nicht, ob
#### wir das am Ende so machen können. Für einen Einzelspalt geht es
####__________________________________________________________________


def interferenz_einzelspalt_analytisch(X,a,wl,zs):
	alphax = arctan(X/zs)
	return (((a*sinc(0.5*a*k(wl)*sin(alphax))))**2)
def interferenz_Nspalt_analytisch(X,n,a,d,wl,zs):
	return_vec = []
	for x in X:
		alphax = arctan(x/zs)
		if n==0:
			return_vec.append(1)
		elif x==0:
			return_vec.append((n*a)**2)
			#return_vec.append((a * sinc(pi*a/wl*sin(alphax)))**2)
		elif sin(pi*d/wl*sin(alphax))==0:
			return_vec.append((n*a*sinc(pi*a/wl*sin(alphax)))**2)
		else:
			return_vec.append((n*sin(pi*n*d/wl*sin(alphax))/(sin(pi*d/wl*sin(alphax))) * a * sinc(pi*a/wl*sin(alphax)))**2)
	return return_vec

def interferenz_Gitter_analytisch(X,Y,nx,ny,ax,ay,dx,dy,wl,zs):
	x_arr = interferenz_Nspalt_analytisch(X,nx,ax,dx,wl,zs)
	y_arr = interferenz_Nspalt_analytisch(Y,ny,ay,dy,wl,zs)
	x_mat, y_mat = np.meshgrid(x_arr,y_arr)
	return x_mat*y_mat

####__________________________________________________________________
#### Hauptfunktionen für n Spalte, Gitter, Gitter mit Fehlstelle etc..
#### Aufzurufen aus der main()
####__________________________________________________________________

def Main_Canvas(wl,zs):
	canvas_size = 500 		#Groesse der quadratischen Leinwand
	drawradius = 10			#Stiftdicke

	# wir legen ein Matrix an (Liste in Liste) mit den Dimenstion canvas_size x canvas_size
	# jeder Eintrag entspricht einem Pixel auf der Canvas. Moegliche Eintraege: 0 fuer keine Transmission, 1 fuer Transmission
	imagearray =  [[ 0 for xcoord in range( canvas_size ) ] for ycoord in range( canvas_size ) ]

	### getNeightbourPixels(int x0, int y0)
	###	Beim Malen auf der Canvas werden die akutellen Mauskoordinaten (x0,y0) uebermittelt. 
	###	Wir berechnen hier nun alle weiteren Pixel im Umkreis von drawradius
	def getNeightbourPixels(x0,y0):
		x0 = int(x0)
		y0 = int(y0)
				
		tempx = x0-drawradius
		tempy = y0-drawradius

		if (tempx < 0) :
			tempx = 0
		if (tempy < 0):
			tempy = 0

		while (tempx <= x0+drawradius) and (tempx < canvas_size):
			tempy = y0-drawradius
			while (tempy < y0+drawradius) and (tempy < canvas_size):
				if (abstandZweierPkte(x0,y0,tempx,tempy) <= drawradius):
					imagearray[tempy][tempx] = 1
				tempy = tempy+1
			tempx = tempx + 1	

		#print("durchlaufx" + str(durchlaufx))
		#print("durchlaufy" + str(durchlaufy))

	def paint( event ):
		draw_color = "#FFFFFF"
		x1, y1 = ( event.x - drawradius ), ( event.y - drawradius)
		x2, y2 = ( event.x + drawradius ), ( event.y + drawradius)
		getNeightbourPixels(event.x,event.y)
		w.create_oval( x1, y1, x2, y2, fill = draw_color, outline=draw_color )

	def drawPlot():
		trans=np.array(imagearray)
		X_trans, Y_trans = np.meshgrid(np.linspace(-trans.shape[1]/2,trans.shape[1]/2,trans.shape[1]), np.linspace(-trans.shape[0]/2,trans.shape[0]/2,trans.shape[0]))
		X,Y,Z = fftCanvas2D_XYZ(np.array(imagearray),wl,zs/30)
		Z /= np.nanmax(Z)

		levels_Z = [0, 1./1000., 1./300., 1./100., 1./30., 1./10., 1./3., 1.]
		cmap_lin = plt.cm.Reds
		cmap_nonlin_Z = nlcmap(cmap_lin, levels_Z)
		
		fig, ax = plt.subplots(nrows=1, ncols=2)
	
		plt.subplot(1,2,1)
		plt.contourf(X_trans,-Y_trans,trans,cmap='gray')
		
		plt.subplot(1,2,2)
		plt.contourf(X,Y,Z,levels=levels_Z,cmap=cmap_nonlin_Z)
		plt.colorbar()
				
		plt.show()	

	master = Tk()
	master.title( "Beugungsmuster" )
	w = Canvas(master, 
			   width=canvas_size, 
			   height=canvas_size,
			   bg="#000000")
	w.pack(expand = NO, fill = BOTH)
	w.bind("<B1-Motion>", paint)
	
	b = Button(master, text="Show Plot", command=drawPlot)
	b.pack()
	message = Label( master, text = "Press and Drag the mouse to draw." )
	message.pack( side = BOTTOM )
	mainloop()	
	
	'''
	### Output imagearray in a better format then print for easy comparism with the drawn picture
	for row in imagearray:
		rowcontent = ""
		for entry in row:
			rowcontent += str(entry)
		print(rowcontent)
	'''


def Main_Default(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
		
	### Init and Parameters for plotting
	resolution = 100
	x1  = np.linspace(-5., 5., resolution)
	y1  = np.linspace(-5., 5., resolution)
	X,Y = np.meshgrid(x1, y1)
	levels_screen = [0, 1./1000., 1./300., 1./100., 1./30., 1./10., 1./3., 1.]
	cmap_lin = plt.cm.Reds
	cmap_nonlin = nlcmap(cmap_lin, levels_screen)

	
	### Objektebene
	x_obj = np.array(np.linspace(-max(nx,ny)*max(dx,dy)/2,max(nx,ny)*max(dx,dy)/2,resolution))
	y_obj = x_obj
	x_obj_mesh, y_obj_mesh = np.meshgrid(x_obj,y_obj)
	intensity_obj       = Transmission_Gitter(x_obj,y_obj,nx,ny,ax,ay,dx,dy,0,error_matrix)
	intensity_obj_error = Transmission_Gitter(x_obj,y_obj,nx,ny,ax,ay,dx,dy,errortype,error_matrix)
	
	### DFT
	start_time_dft = time.time()
	intensity_DFT  = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
	intensity_DFT /= intensity_DFT.max() #normierung
	total_time_dft = formatSecToMillisec(time.time() - start_time_dft)
	print("DFT Berechnung dauerte: " + total_time_dft)

	if errortype == 0:
		### Analyisch
		start_time_anal = time.time()
		intensity_anal = interferenz_Gitter_analytisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
		intensity_anal /= intensity_anal.max()
		total_time_anal = formatSecToMillisec(time.time() - start_time_anal)
		print("Analytische Berechnung dauerte: " + total_time_anal)
	

	### FFT
	start_time_fft = time.time()
	XX, YY, intensity_fft = fftNspalt2D_XYZ(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
	intensity_fft/=intensity_fft.max()
	total_time_fft =  formatSecToMillisec(time.time() - start_time_fft)
	print("FFT Berechnung dauerte: " + total_time_fft)
	
	plt.figure(0)

	if errortype == 0:
		# Die drei Objektebenen, die ersten beiden ohne Fehler (da analytisch und dft), der dritte mit Fehler (da fft)
		plt.subplot2grid((2, 3), (0, 0))
		plt.pcolor(x_obj_mesh*1000000, y_obj_mesh*1000000,intensity_obj, cmap='gray')
		
		plt.subplot2grid((2, 3), (0, 1))
		plt.pcolor(x_obj_mesh*1000000, y_obj_mesh*1000000,intensity_obj_error, cmap='gray')
		
		plt.subplot2grid((2, 3), (0, 2))
		plt.pcolor(x_obj_mesh*1000000, y_obj_mesh*1000000,intensity_obj_error, cmap='gray')
		

		plt.subplot2grid((2, 3), (1, 0))
		plt.subplot2grid((2, 3), (1, 0)).set_title("Analytisch. t=" + total_time_anal )
		plt.contourf(X,Y,intensity_anal,levels=levels_screen,cmap=cmap_nonlin)
		plt.colorbar()

		plt.subplot2grid((2, 3), (1, 1))
		plt.subplot2grid((2, 3), (1, 1)).set_title("DFT. t=" + total_time_dft)
		plt.contourf(X,Y,intensity_DFT,levels=levels_screen,cmap=cmap_nonlin)
		plt.colorbar()

		
		plt.subplot2grid((2, 3), (1, 2))  
		plt.subplot2grid((2, 3), (1, 2)).set_title("FFT. t=" +total_time_fft)
		plt.contourf(XX,YY,intensity_fft,levels=levels_screen,cmap=cmap_nonlin)
		plt.colorbar()

	# errortype != 0, thus showing only fft
	else:
		plt.subplot2grid((1,3), (0, 0))
		plt.pcolor(x_obj_mesh*1000000, y_obj_mesh*1000000,intensity_obj_error, cmap='gray')

		plt.subplot2grid((1, 3), (0, 1))
		plt.subplot2grid((1, 3), (0, 1)).set_title("DFT. t=" + total_time_dft)
		plt.contourf(X,Y,intensity_DFT,levels=levels_screen,cmap=cmap_nonlin)
		plt.colorbar()

		plt.subplot2grid((1,3), (0, 2))
		plt.subplot2grid((1,3), (0, 2)).set_title("FFT. t=" +total_time_fft)
		plt.contourf(XX,YY,intensity_fft,levels=levels_screen,cmap=cmap_nonlin)
		plt.colorbar()
	
	plt.suptitle('Breite x (um): '+ str(round(ax*1e6)) + ', Hoehe y (um): '+str(round(ay*1e6)) + ', Abstand in x (um):' + str(round(dx*1e6)) + ', Abstand in y (um):' + str(round(dy*1e6)) + ', Wellenlänge in nm:' + str(round(wl*1e9))  + ', Schirmabstand in m: ' + str(zs))  
	plt.show()
	

### execute main() when program is called	
if __name__ == "__main__":
	main()