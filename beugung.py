#!/usr/bin/env python
# -*- coding: utf-8; -*-

# 2017/2018 
# Authors: Bernd Lienau, Simon Jung, Alexander Franke
# published under GNU General Public License v3.0
# see LICENSE file for further details


# Import Math functions
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
import math
import scipy
from scipy import integrate as integrate

# Import Matplotlib for output
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pylab import *
import matplotlib.pyplot as plt

# Misc Imports
import sys
import time
import random
import pandas as pd
import argparse

# Input Canvas
from tkinter import *

####### TODOS ######
'''
moeglich:
	- Uberlagerung von verschiendenen Wellenlaengen
	- mehrere Gitter hintereinander
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
	parser.add_argument('--errortype', dest='errortype', help='Gitterfehlertyp. 0 fuer keinen Fehler. 1 fuer zufaellige, kleine Verschiebung jedes Spaltes, 2 fuer 15% Chance fuer jedes Loch, dass es nicht existiert (Fehlstellen)',default=0)
	parser.add_argument('--wl', dest='wl',help='Wellenlaenge in nm',default=780 )
	parser.add_argument('--zs', dest='zs', help='Schirmabstand in cm',default=350)
	parser.add_argument('--dft', dest='dft', help='Wenn dieser Wert auf true gesetzt wird, wird zusaetzlich eine DFT durchgefuehrt, die unter Umstaenden sehr lange dauern kann. Der Defaultwert ist false.',default='false')
	parser.add_argument('--canvas', dest='canvas',help='Wird dieser Wert auf true gesetzt, wird die Moeglichkeit gegeben anstelle eines Gitter eine Freihandzeichnung zu erstellen. Die meisten anderen Parameter sind dann unerheblich.',default='false')
	
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
	print("   Wellenlaenge in  nm     " + str(args.wl)                                     )
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
	dft = str(args.dft)
	errortype = int(args.errortype)
	wl = int(args.wl) * 1e-9
	zs = int(args.zs) * 1e-2
	canvas = str(args.canvas)
	
	
	matplotlib.rcParams.update({'font.size': 20}) ## change font size

	
	if nx==0 and ny==0:
		print('Ohne Gitter gibt es keine Berechnung...')
		sys.exit(0)

	if nx==0 and ny!=0:
		nx = ny
		ny = 0


	if canvas == 'true': 
		# Leinwand aehnlich zu mspaint. Beugungsmuster eines beliebigen Objekts
		Main_Canvas(wl,zs)
	elif canvas == 'false':
		Main_Default(nx,ny,ax,ay,dx,dy,errortype,Err_matrix_init(nx,ny,errortype),wl,zs,dft)
	

	#__________________________________________________________________
	# Ende der main()



### Err_matrix_init(int nx, int ny, int errortype):
### return if type 1: Transformiert ein fehlerfreies Gitter definiert durch die Anzahl der Spalte in x- und y-Richtung nx und ny in ein Gitter mit leicht verschobenen Loechern
### return if type 2: Transformiert ein fehlerfreies Gitter definiert durch die Anzahl der Spalte in x- und y-Richtung in eines mit Fehlstellen

def Err_matrix_init(nx,ny,errortype):
	if (errortype ==2):
		matrix = []
		for i in range(ny):
			error_row=[]
			for j in range(nx):
				if(random.randint(0, 100) > 15):
					error_row.append([[1,1],[1,1]])
				else:
					error_row.append([[0,0],[0,0]])
			matrix.append(error_row)
	else: #error type 1
		matrix = []
		for i in range(ny):
			error_row=[]
			for j in range(nx):
				error_row.append([[random.uniform(-0.2,0.2),random.uniform(0.9,1.1)],[random.uniform(-0.2,0.2),random.uniform(0.9,1.1)]])
			matrix.append(error_row)
		#falls ny==0 sollen die spalte unendlich lang sein. Fehler ist nicht möglich
		#if ny==0 or nx==0:
		#	matrix.append([[[0,0],[0,0]]])

	return np.array(matrix)


####__________________________________________________________________
#### Hilfsvariablen/funktionen. 
####__________________________________________________________________

def NSpalt_Mittelpunkte(n,d):
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

### kgV2(int a, int b): 
### returns: KgV der beiden ints a und b zurueck
def kgV2(a, b):
	return (a * b) // math.gcd(a, b)

### kgV_arr([int] numbers): 
### returns: kgV einer beliebigen Liste von Ints zurueck
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

### abstandZweierPkte(float x0, float y0, float x, float y)
### returns: Abstand des Punktes (x,y) zum Punkt (x0,y0) auf einer 2D Ebene
def abstandZweierPkte(x0,y0,x,y):
	return round(math.sqrt((x-x0)**2 + (y-y0)**2))

### complex_int(lambda y: func, float a, float b):
### returns: complex integral of func from a to b
def complex_int(func, a, b, **kwargs):
	def real_func(x):
		return scipy.real(func(x))
	def imag_func(x):
		return scipy.imag(func(x))
	real_integral = integrate.quad(real_func, a, b, **kwargs)
	imag_integral = integrate.quad(imag_func, a, b, **kwargs)
	return (real_integral[0] + i()*imag_integral[0])




### formatSecToMillisec(float time)
### returns: converts entered sec time to ms, rounds and adds Einheit
def formatSecToMillisec(time):
	return str(round(time*1000)) + ' ms'

### class nlcmap()
### zur Darstellung der Intensitaet verwenden wir einen nicht-linearen, diskreten Farbverlauf nach 
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

def DFT_2D_Periodisch(xArray,yArray,nx,ny,ax,ay,dx,dy,wl,zs):  
	#Nutzt aus, dass wir die Fouriertransformation vom Einzelspalt kennen, anstatt sie auszurechnen. 
	#Ist also eine Kombination aus analytischem Ergebnis und DFT.
	#Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
	#beim Integrieren bzw bei der fft schief geht.
	subArrayX= []
	subArrayY= []
	
	for x in xArray:
		if nx==0:
			subArrayX.append(1)
		else:
			subArrayX.append(float(DFT_1D_Periodisch_Integrate(x,nx,ax,dx,wl,zs)))
	for y in yArray:
		if ny==0:
			subArrayY.append(1)
		else:
			subArrayY.append(float(DFT_1D_Periodisch_Integrate(y,ny,ay,dy,wl,zs)))
		
	XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
	Ztmp=XX*YY

	return Ztmp

def DFT_1D_Periodisch_Integrate(x,n,a,d,wl,zs):
	u = k(wl)*sin(arctan(x/zs))
	f = lambda y:  Trans_1Spalt(y,a) *exp(-i()*u*y) 
	r = 0
	
	mittelpunkteDerLoecher = NSpalt_Mittelpunkte(n,d)
	for pkt in mittelpunkteDerLoecher:
		r = r + (exp(i()*u*pkt))

	if(n==1):
		r = 1
	integral = complex_int(f,-a,a)
	integral =  scipy.real(np.square(n * np.multiply(integral,r)))
	return integral

def DFT_2D_Any(xArray,yArray,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
	## bietet die Moeglichkeit beliebige Funktion fuer das Gitter einzusetzen
	
	#Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
	#beim Integrieren bzw bei der fft schief geht.

	Ztotal=[]
	subArrayX=[]
	
	for y in yArray:
		for x in xArray:
			subArrayX.append(float(DFT_2D_Any_Integrate(x,y,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)))
		Ztotal.append(subArrayX)
		subArrayX=[]
		
	return np.array(Ztotal)


def DFT_2D_Any_Integrate(xSchirm,ySchirm,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
	# Fouriertransformierte von Trans_Gitter
	
	## bietet die Moeglichkeit eine beliebige Funktion fuer das Gitter in 'Trans_NSpalt(y,n,a,d)' einzusetzen
	
	u = k(wl)*sin(arctan(xSchirm/zs))
	v = k(wl)*sin(arctan(ySchirm/zs))
	def DFT_2D_Any_Integration_Function(y,x,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs,u,v):
		tr = Trans_Gitter_float(x,y,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
		if ( tr != 0):
			return tr * exp(complex(0,-1)*(u*x+v*y))
		else: 
			return 0


	integral,error = integrate.dblquad(DFT_2D_Any_Integration_Function,-(ny-1)*dy/2-ay,(ny-1)*dy/2+ay , lambda x: -(nx-1)*dx/2-ax, lambda x: (nx-1)*dx/2+ax, args=(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs,u,v))
	integral = np.square(integral)
	
	return integral

def FFT_1D(nx,ax,dx,errortype,error_array,wl,zs):
	datapoints = kgV_arr([int(dx*1e6*2*nx),int(ax*1e6)]) ## minimale Anzahl an Datenpunkten, damit an jedem Spaltrand ein Punkt liegt
	while(datapoints*wl/(4*nx*dx) < 0.82):                 ## erhoehe Datapoints, damit mindestens die Raumfrequenzen berechnet werden, die auf dem Schirm abgebildet werden
		datapoints*=2                                     ## 0.82 fuer Plot bis +-5m
	datapoints = datapoints +1                                    ## Datenpunkte im ganzen Array, mit Anfang- und Endpunkt, daher +1
	
	x_Spalt = np.array(np.linspace(-dx*nx,dx*nx,datapoints)) ## waehle großen Bereich fuer die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist
	
	deltax = (x_Spalt[1]-x_Spalt[0])
	fa = 1.0/deltax #Nyquist-Frequenz
	Xf = tan(arcsin(np.linspace(-fa/2,fa/2,datapoints)*wl))*zs  #Datenpunkte der fft als k-Vektoren im np.linspace(..)
	# zurueckgerechnet in x-/y-Positionen auf dem Schirm via Gl. LS(k) = integral(transmission(x)*exp(-2*pi*i*k*x)dx)
	# hierbei ist k die Wellenzahl und somit gibt LS(k)/k0=LS(k)*wl=sin(alphax) den Winkel zur Stahlachse an,
	# unter dem der gebeugte Strahl propagiert. Mit Hilfe des tan(alphax) und der Schirmentfernung zs findet sich
	# so durch tan(alphax)*wl=tan(arcsin(LS(k)*wl))*zs die x-Koordinate auf dem Schirm, zu der der k-Vektor der fft gehoert.
	# So wird Xf berechnet, welches jedem Intensitaetswert aus der fft den richtigen Punkt auf dem Schirm zuordnet
	
	index_low = np.argmax(Xf>-5.0) #Beschraenke den Plot auf -5m bis +5m auf dem Screen
	index_high = np.argmax(Xf>5.0)
	if index_high==0:
		index_high=len(Xf)
	X_Schirm = Xf[index_low:index_high]
	
	## 1D Berechnung
	Transmission_X = []
	for x in x_Spalt:
		Transmission_X.append(Trans_NSpalt(x,nx,ax,dx,errortype,error_array))
	z1Df = fftshift(np.square(np.abs(fft(Transmission_X))*2/datapoints))
		
	return X_Schirm, z1Df[index_low:index_high]

def FFT_2D(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs,schirmgroesse): ##Ergebnisform stimmt, Skalierung noch nicht
		
	
	if nx==0:  ## then only integrate in y-direction
		Schirm, z1Dfy = FFT_1D(ny,ay,dy,errortype,error_matrix[:,0,1],wl,zs) ## error_matrix[:,0,1] is the array of the error-values in y-direction
		z1Df_X, z1Df_Y = np.meshgrid(np.ones(len(z1Dfy)),z1Dfy)
		z2Df = z1Df_X * z1Df_Y
		X_Schirm, Y_Schirm = np.meshgrid(Schirm,Schirm)
	elif ny==0: ## then only integrate in x-direction
		Schirm, z1Dfx = FFT_1D(nx,ax,dx,errortype,error_matrix[0,:,0],wl,zs) ## error_matrix[0,:,0] is the array of the error-values in x-direction
		z1Df_X, z1Df_Y = np.meshgrid(z1Dfx,np.ones(len(z1Dfx)))
		z2Df = z1Df_X * z1Df_Y
		X_Schirm, Y_Schirm = np.meshgrid(Schirm,Schirm)
	else:
		## 2D Berechnung
		d = max(dx,dy)
		n = max(nx,ny)
		a = max(ax,ay)
		datapoints = kgV_arr([int(d*1e6*2*n),int(a*1e6)])  ## minimale Anzahl an Datenpunkten, damit an jedem Spaltrand ein Punkt liegt
		while(datapoints*wl/(4*n*d) < (0.82/5)*schirmgroesse):                 ## erhoehe Datapoints, damit mindestens die Raumfrequenzen berechnet werden, die auf dem Schirm abgebildet werden
			datapoints*=2                                   ## 0.82*2 fuer Plot bis +-5m
		datapoints = int(datapoints)+1                                   ## Datenpunkte im ganzen Array, mit Anfang- und Endpunkt, daher +1
		x_Spalt = np.array(np.linspace(-n*d,n*d,datapoints))   
		y_Spalt = np.array(np.linspace(-n*d,n*d,datapoints))   

		deltax = (x_Spalt[1]-x_Spalt[0]) #Sampling-Rate ist fuer x- und y-Richtung gleich
		fa = 1.0/deltax #Nyquist-Frequenz
		Xf = tan(arcsin(np.linspace(-fa/2,fa/2,datapoints)*wl))*zs  #Datenpunkte der fft als k-Vektoren im np.linspace(..)
		# zurueckgerechnet in x-/y-Positionen auf dem Schirm via Gl. LS(k) = integral(transmission(x)*exp(-2*pi*i*k*x)dx)
		# hierbei ist k die Wellenzahl und somit gibt LS(k)/k0=LS(k)*wl=sin(alphax) den Winkel zur Stahlachse an,
		# unter dem der gebeugte Strahl probagiert. Mit Hilfe des tan(alphax) und der Schirmentfernung zs findet sich
		# so durch tan(alphax)*wl=tan(arcsin(LS(k)*wl))*zs die x-Koordinate auf dem Schirm, zu der der k-Vektor der fft gehoert.
		# So wird Xf berechnet, welches jedem Intensitaetswert aus der fft den richtigen Punkt auf dem Schirm zuordnet
		Yf = tan(arcsin(np.linspace(-fa/2,fa/2,datapoints)*wl))*zs

		index_low =  np.argmax(Xf>-schirmgroesse) #Beschraenke den Plot auf -5m bis +5m auf dem Screen
		index_high = np.argmax(Xf>schirmgroesse)
		if index_high==0:
			index_high = len(Xf)
		X_Schirm, Y_Schirm = np.meshgrid(Xf[index_low:index_high],Yf[index_low:index_high])
		
		z2D = Trans_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,errortype,error_matrix)
		z2Df = fftshift(np.square(np.abs(fft2(z2D))*4/(datapoints**2)))[index_low:index_high,index_low:index_high]
	return X_Schirm, Y_Schirm, z2Df

def fourierNspaltIntegrateAnyFunction(xSchirm,n,a,d,errortype,error_array,wl,zs):
	# Fouriertransformierte von Trans_Gitter
	
	## bietet die Moeglichkeit eine beliebige Funktion fuer das Gitter in 'Trans_NSpalt(y,n,a,d)' einzusetzen
	
	u = k(wl)*sin(arctan(xSchirm/zs))
	#lambda x sagt python nur dass das die Variable ist und nach der integriert werden muss
	f = lambda y:  Trans_NSpalt(y,n,a,d,errortype,error_array)*exp(-i()*u*y) 

	integral = complex_int(f,-(n-1)*d/2-a,(n-1)*d/2+a)
	#scipy.real koennte man weg lassen, da korrekterweise der imaginaer Teil immer null ist. Aber damit
	#matplot keine Warnung ausgibt, schmeissen wir den img Teil hier weg.
	integral =  scipy.real(np.square(np.multiply(n,integral)))
	return integral

def FFT_2D_Canvas(imagearray,wl,zs): ## 1 pixel = 0.1 um
	## 2D Berechnung
	N = 800 ## Datenpunkte im ganzen Array, mit Anfang- und Endpunkt, daher +1
	x_Spalt = np.array(np.linspace(-N/2*1e-7,N/2*1e-7,N))   ## waehle großen Bereich fuer die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist
	y_Spalt = np.array(np.linspace(-N/2*1e-7,N/2*1e-7,N))   ## waehle großen Bereich fuer die Transmissionsfunktion, damit die x-Skalierung nach der fft feiner ist
	
	z2D = np.hstack((np.zeros(shape=(imagearray.shape[0], int((N-imagearray.shape[1])/2))), imagearray,
					 np.zeros(shape=(imagearray.shape[0], int((N-imagearray.shape[1])/2)))))
	z2D = np.vstack((np.zeros(shape=(int((N-z2D.shape[0])/2),int(z2D.shape[1]))), z2D,
					 np.zeros(shape=(int((N-z2D.shape[0])/2),int(z2D.shape[1])))))
	
	deltax = (x_Spalt[1]-x_Spalt[0]) #Sampling-Rate ist fuer x- und y-Richtung gleich
	fa = 1.0/deltax #Nyquist-Frequenz
	Xf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs  #Datenpunkte der fft als k-Vektoren im np.linspace(..)
	# zurueckgerechnet in x-/y-Positionen auf dem Schirm via Gl. LS(k) = integral(transmission(x)*exp(-2*pi*i*k*x)dx)
	# hierbei ist k die Wellenzahl und somit gibt LS(k)/k0=LS(k)*wl=sin(alphax) den Winkel zur Stahlachse an,
	# unter dem der gebeugte Strahl probagiert. Mit Hilfe des tan(alphax) und der Schirmentfernung zs findet sich
	# so durch tan(alphax)*wl=tan(arcsin(LS(k)*wl))*zs die x-Koordinate auf dem Schirm, zu der der k-Vektor der fft gehoert.
	# So wird Xf berechnet, welches jedem Intensitaetswert aus der fft den richtigen Punkt auf dem Schirm zuordnet
	Yf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs

	index_low =  np.argmax(Xf>-5.0) #Beschraenke den Plot auf -5m bis +5m auf dem Screen
	index_high = np.argmax(Xf>5.0)
	if index_high==0:
		index_high = len(Xf)
	X_Schirm, Y_Schirm = np.meshgrid(Xf,Yf)#[index_low:index_high],Yf[index_low:index_high])
	
	z_fft = fft2(z2D)

	z2Df = fftshift(np.square(np.abs(z_fft))) #[index_low:index_high,index_low:index_high]
	#Ruecktransformation
	z2Df_back = (np.square(np.abs(fft2(z_fft)))) #[index_low:index_high,index_low:index_high]

	return X_Schirm, Y_Schirm, z2Df, z2Df_back


####__________________________________________________________________
#### Transmissionsfunktion verschiedener Objekte
####__________________________________________________________________

def Trans_1Spalt(x,a):
	if math.fabs(x) <= a/2:
		return 1
	else:
		return 0

def Trans_Loch(rho,R):
	#einzelnes Loch mit Radius R
	#Verwende Polarkoordinaten rho,theta 
	if (rho <= R):
		return 1
	else: 
		return 0
	

def Trans_NSpalt(x,n,a,d,errortype,error_array):
	gesamttransmission = 0.
	
	if errortype == 0:  ##kein Error
		for i in range(n):
			gesamttransmission += Trans_1Spalt(x-d*(i-n/2+0.5),a)
	
	elif errortype == 1:
		for i in range(n):
			gesamttransmission += Trans_1Spalt(x+error_array[i,0]*a-d*(i-n/2+0.5),a*error_array[i,1])
			
	elif errortype == 2:
		#there is a 15% chance that an hole is missing
		for i in range(n):
			gesamttransmission += Trans_1Spalt(x-d*(i-n/2+0.5),a) * int(error_array[i,0])

	return gesamttransmission

def Trans_Gitter_float(x,y,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
	
	trans=0.0
	for i in range(max(nx,1)):
		for j in range(max(ny,1)):
			if errortype==0:
				trans+=Trans_1Spalt(x-dx*(i-nx/2+0.5),ax)*Trans_1Spalt(y-dy*(j-ny/2+0.5),ay)
			elif errortype==1:
				# soll jede einzelne Koordinate (x,y) nehmen und schauen, ob dort Durchlass (0,1) ist
				# Transmission in x-Richtung am Punkt (x,y) für alle Spalte (nx,ny)       *  Transmission in y-Richtung am Punkt (x,y) für alle Spalte (nx,ny)
				trans+=Trans_1Spalt(x+error_matrix[j,i,0,0]*ax-dx*(i-nx/2+0.5),ax*error_matrix[j,i,0,1])*Trans_1Spalt(y+error_matrix[j,i,1,0]*ay-dy*(j-ny/2+0.5),ay*error_matrix[j,i,1,1])
			elif errortype==2:
				trans+=Trans_1Spalt(x+error_matrix[j,i,0,0]*ax-dx*(i-nx/2+0.5),ax*error_matrix[j,i,0,1])*Trans_1Spalt(y+error_matrix[j,i,1,0]*ay-dy*(j-ny/2+0.5),ay*error_matrix[j,i,1,1])
	return trans
	
def Trans_Gitter(xArray,yArray,nx,ny,ax,ay,dx,dy,errortype,error_matrix):
	# Returns the transmission for a periodic grid as a matrix with 0/1
	# to plot it with the contourplot-fct or use it for the fft2 algorithm
	# error is the error of the grid which is given to the Transmission-fct
	subArrayX=[]
	subArrayY=[]
	
	## Teile xArray an Mittelpunkten zwischen den Spalten in Teile von
	##     [:-(nx-2)/2*dx][-(nx-2)/2*dx:(i-(nx-2)/2*dx)][...] i in range(nx-1)
	## Teile yArray an Mittelpunkten zwischen den Spalten in Teile von
	##     [:-(ny-2)/2*dy][-(ny-2)/2*dy:(j-(ny-2)/2*dy)][...] j in range(ny-1)
	## Erhalte somit nx*ny Teilstuecke des Gitters, in denen sich jeweils ein Spalt befindet
	## Integriere fuer jeden einzelnen Spalt separat, fuelle die restlichen Gebiete des Ergebnisses mit 1
	## multipliziere die einzelnen Spaltfouriertransformierten um das Gesamtergebnis zu erhalten
	Ztmp=[]

	for i in range(max(nx,1)):
		for j in range(max(ny,1)):
			for x in xArray:
				if nx==0:
					subArrayX.append(1)
				elif x > ((i-1-(nx-2)/2)*dx) and x <= ((i-(nx-2)/2)*dx):
					subArrayX.append(Trans_NSpalt(x,nx,ax,dx,errortype,error_matrix[j,:,0]))
				else:
					subArrayX.append(0)
			for y in yArray:
				if ny==0:
					subArrayY.append(1)
				elif y > ((j-1-(ny-2)/2)*dy) and y <= ((j-(ny-2)/2)*dy):
					subArrayY.append(Trans_NSpalt(y,ny,ay,dy,errortype,error_matrix[:,i,0]))
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
#### Intensitaetsverteilungen fuer verschiedene Objekte. Ich weiß nicht, ob
#### wir das am Ende so machen koennen. Fuer einen Einzelspalt geht es
####__________________________________________________________________

def Calc_Ana_2Spalt(x,a,d,wl,zs):
	sinalphax = sin(arctan(x/zs))
	return cos(pi*d*sinalphax/ (wl))**2 * (sin(pi*a*sinalphax/wl)**2) / ((pi*a*sinalphax/wl)**2)

def Calc_Ana_NSpalt(X,n,a,d,wl,zs):
	return_vec = []
	for x in X:
		alphax = arctan(x/zs)
		if n==0:
			return_vec.append(1)
		elif n==2:
			return_vec.append(Calc_Ana_2Spalt(x,a,d,wl,zs))
		elif x==0:
			return_vec.append((n*a)**2)
			#return_vec.append((a * sinc(pi*a/wl*sin(alphax)))**2)
		elif sin(pi*d/wl*sin(alphax))==0:
			return_vec.append((n*a*sinc(pi*a/wl*sin(alphax)))**2)
		else:
			return_vec.append((n*sin(pi*n*d/wl*sin(alphax))/(sin(pi*d/wl*sin(alphax))) * a * sinc(pi*a/wl*sin(alphax)))**2)
	return return_vec

def Calc_Ana_Gitter(X,Y,nx,ny,ax,ay,dx,dy,wl,zs):
	x_arr = Calc_Ana_NSpalt(X,nx,ax,dx,wl,zs)
	y_arr = Calc_Ana_NSpalt(Y,ny,ay,dy,wl,zs)
	x_mat, y_mat = np.meshgrid(x_arr,y_arr)
	return x_mat*y_mat

####__________________________________________________________________
#### Hauptfunktionen fuer n Spalte, Gitter, Gitter mit Fehlstelle etc..
#### Aufzurufen aus der main()
####__________________________________________________________________

def Main_Canvas(wl,zs):
	canvas_size = 800 		#Groesse der quadratischen Leinwand
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

	def paint( event ):
		draw_color = "#FFFFFF"
		x1, y1 = ( event.x - drawradius ), ( event.y - drawradius)
		x2, y2 = ( event.x + drawradius ), ( event.y + drawradius)
		getNeightbourPixels(event.x,event.y)
		w.create_oval( x1, y1, x2, y2, fill = draw_color, outline=draw_color )

	def drawPlot():
		trans=np.array(imagearray)
		X_trans, Y_trans = np.meshgrid(np.linspace(-trans.shape[1]/2,trans.shape[1]/2,trans.shape[1]), np.linspace(-trans.shape[0]/2,trans.shape[0]/2,trans.shape[0]))
		X,Y,Z, Z_back = FFT_2D_Canvas(np.array(imagearray),wl,zs/30)

		#Ruecktransformation


		Z /= np.nanmax(Z)

		levels_Z = [0, 1./1000., 1./300., 1./100., 1./30., 1./10., 1./3., 1.]
		cmap_lin = plt.cm.Reds
		cmap_nonlin_Z = nlcmap(cmap_lin, levels_Z)
		
		fig, ax = plt.subplots(nrows=1, ncols=3)
	
		plt.subplot(1,3,1)
		plt.subplot(1,3,1).set_title("Objektebene")
		plt.contourf(X_trans,-Y_trans,trans,cmap='gray')

		
		plt.subplot(1,3,2)
		plt.subplot(1,3,2).set_title("Schirm (FFT)")
		plt.contourf(X,Y,Z,levels=levels_Z,cmap=cmap_nonlin_Z)
		plt.colorbar()

		plt.subplot(1,3,3)
		plt.subplot(1,3,3).set_title("Ruecktransformation")
		plt.contourf(-X_trans,Y_trans,Z_back,cmap='gray')
				
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


def Main_Default(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs,dft):
		
	### Init and Parameters for plotting
	resolution = 100
	schirmgroesse = 5.0
	x1  = np.linspace(-schirmgroesse, schirmgroesse, resolution)
	y1  = np.linspace(-schirmgroesse, schirmgroesse, resolution)
	X,Y = np.meshgrid(x1, y1)
	levels_screen = [0, 1./800., 1./300., 1./100., 1./30., 1./10., 1./3., 1.]
	cmap_lin = plt.cm.Reds
	cmap_nonlin = nlcmap(cmap_lin, levels_screen)

	
	### Objektebene
	x_obj = np.array(np.linspace(-max(nx,ny)*max(dx,dy)/2,max(nx,ny)*max(dx,dy)/2,resolution))
	y_obj = x_obj
	x_obj_mesh, y_obj_mesh = np.meshgrid(x_obj,y_obj)
	intensity_obj       = Trans_Gitter(x_obj,y_obj,nx,ny,ax,ay,dx,dy,0,error_matrix)
	intensity_obj_error = Trans_Gitter(x_obj,y_obj,nx,ny,ax,ay,dx,dy,errortype,error_matrix)
	
	### Analyisch
	start_time_anal = time.time()
	intensity_anal = Calc_Ana_Gitter(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
	intensity_anal /= intensity_anal.max()
	total_time_anal = formatSecToMillisec(time.time() - start_time_anal)
	print("Analytische Berechnung dauerte: " + total_time_anal)
	

	if dft == "true":
		### DFT
		start_time_dft = time.time()
		#intensity_DFT  = DFT_2D_Periodisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
		#   DFT_2D_Periodisch is a faster algorithm for a grid without errors, using symmetries and the known result of
		#   the fourier transformation of multiple slits compared to a single slit and thus being
		#   much faster
		intensity_DFT  = DFT_2D_Any(x1,y1,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
		intensity_DFT /= intensity_DFT.max() #normierung
		total_time_dft = formatSecToMillisec(time.time() - start_time_dft)
		print("DFT Berechnung dauerte: " + total_time_dft)

	
	### FFT
	start_time_fft = time.time()
	XX, YY, intensity_fft = FFT_2D(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs,schirmgroesse)
	intensity_fft/=intensity_fft.max()
	total_time_fft =  formatSecToMillisec(time.time() - start_time_fft)
	print("FFT Berechnung dauerte: " + total_time_fft)
	
	plt.figure(0)

	# Die drei Objektebenen, die analytische Berechnung ohne Gitterfehler
	plt.subplot2grid((2, 3), (0, 0))
	plt.pcolor(x_obj_mesh*1000000, y_obj_mesh*1000000,intensity_obj, cmap='gray')
	
	plt.subplot2grid((2, 3), (0, 1))
	plt.pcolor(x_obj_mesh*1000000, y_obj_mesh*1000000,intensity_obj_error, cmap='gray')
	
	plt.subplot2grid((2, 3), (0, 2))
	plt.pcolor(x_obj_mesh*1000000, y_obj_mesh*1000000,intensity_obj_error, cmap='gray')
	

	#Schirm

	#Analytisch
	plt.subplot2grid((2, 3), (1, 0))
	plt.subplot2grid((2, 3), (1, 0)).set_title("Analytisch. t=" + total_time_anal )
	plt.contourf(X,Y,intensity_anal,levels=levels_screen,cmap=cmap_nonlin)
	plt.colorbar()

	#DFT
	if dft == "true":
		plt.subplot2grid((2, 3), (1, 1))
		plt.subplot2grid((2, 3), (1, 1)).set_title("DFT. t=" + total_time_dft)
		plt.contourf(X,Y,intensity_DFT,levels=levels_screen,cmap=cmap_nonlin)
		plt.colorbar()
	else:
		plt.subplot2grid((2, 3), (1, 1))
		plt.subplot2grid((2, 3), (1, 1)).set_title("DFT not calculated")

	#FFT
	plt.subplot2grid((2, 3), (1, 2))  
	plt.subplot2grid((2, 3), (1, 2)).set_title("FFT. t=" +total_time_fft)
	plt.contourf(XX,YY,intensity_fft,levels=levels_screen,cmap=cmap_nonlin)
	plt.colorbar()

	plt.suptitle('Breite x (um): '+ str(round(ax*1e6)) + ', Hoehe y (um): '+str(round(ay*1e6)) + ', Abstand in x (um):' + str(round(dx*1e6)) + ', Abstand in y (um):' + str(round(dy*1e6)) + ', Wellenlaenge in nm:' + str(round(wl*1e9))  + ', Schirmabstand in m: ' + str(zs))  
	plt.show()
	

### execute main() when program is called	
if __name__ == "__main__":
	main()