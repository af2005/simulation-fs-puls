#!/usr/bin/env python
# -*- coding: utf-8; -*-
#
# Copyright (C) 2017 Bernd Lienau, Simon Jung, Alexander Franke

import math
import cmath
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

from matplotlib.colors import LinearSegmentedColormap
from pylab import *
 

import csv
import pandas as pd
import scipy
from scipy import integrate as integrate


import matplotlib.pyplot as plt
import argparse

## zur Darstellung der Intensität verwenden wir einen nicht-linearen Farbverlauf nach 
## https://stackoverflow.com/questions/33873397/nonlinear-colormap-with-matplotlib
class nlcmap(LinearSegmentedColormap):
	"""A nonlinear colormap"""

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


#main() wird automatisch aufgerufen
def main():
	parser = argparse.ArgumentParser(description='This is a python3 module simulating a light pulse with given parameters propagating through different optical components suchas Einzelspalt, Doppelspalt, Gitter mit und ohne Fehlstellen oder Defekten.')
	#
	# Für Spalte nur in 1 Richtung mit unendlicher Ausdehnung einfach nx/ny auf 0 setzen
	#parser.add_argument('--dimension', dest='dimension',help='Auf 1 zu setzen für n Spalte, auf 2 für Gitter .',default=2)
	#
	parser.add_argument('--nx', dest='nx', help='Die Anzahl der Spalte in x-Richtung. Ganzzahlige Zahl zwischen 0 und Unendlich. 0 steht hierbei fuer einen Spalt mit unendlicher Ausdehnung.',default=1)
	parser.add_argument('--ny', dest='ny', help='Die Anzahl der Spalte in x-Richtung. Ganzzahlige Zahl zwischen 0 und Unendlich. 0 steht hierbei fuer einen Spalt mit unendlicher Ausdehnung.',default=1)
	parser.add_argument('--ax', dest='ax', help='Spaltbreite in um',default=3)
	parser.add_argument('--ay', dest='ay', help='Spalthoehe in um',default=5)
	parser.add_argument('--dx', dest='dx', help='Spaltabstand in horizontaler Richtung in um',default=10)
	parser.add_argument('--dy', dest='dy', help='Spaltabstand in vertikaler Richtung in um',default=10)
	parser.add_argument('--error', dest='error', help='Gitterfehler in um, verschiebt momentan die Spaltmittelpunkte um diesen Wert in x- und y-Richtung von (0,0) weg',default=1)
	parser.add_argument('--wl', dest='wl',help='Wellenlänge in nm',default=780 )
	parser.add_argument('--abstand', dest='zs', help='Schirmabstand in cm',default=350)
	


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
	print('Es wurden folgende Parameter eingegeben oder angenommen'                           )
	#print('   Dimension                                  :  ' + str(args.dimension))
	#print('   Falls Dimension = 1, berechnen wir für     :')
	print('   Anzahl der Spalte:')
	print('      ' + str(args.nx) + ' Spalte in horizontaler Richtung(x)')
	print('      ' + str(args.ny) + ' Spalte in vertikaler Richtung(y)')
	print('   Spaltbreite in um          :')
	print('      horizontal(x):   ' + str(args.ax))
	print('      vertikal(y):     ' + str(args.ay))
	print('   Spaltabstand in um          :')
	print('      horizontal(x):   ' + str(args.dx))
	print('      vertikal(y):     ' + str(args.dy))
	print('   Gitterfehler:       ' + str(args.error) + 'um Verschiebung aller Spalte in x- und y-Richtung vom Zentrum (0,0) weg')
	print("   Wellenlänge in  nm                         :  " + str(args.wl))
	print("   Schirmabstand in cm                        :  " + str(args.zs))
	print('')
	print('------------------------------------------------------------------------------')

	#__________________________________________________________________
	# Variablen auf SI Einheiten bringen. 
	nx = int(args.nx)
	ny = int(args.ny)
	ax = int(args.ax)  * 1e-6
	ay = int(args.ay)  * 1e-6
	dx = int(args.dx)  * 1e-6
	dy = int(args.dy)  * 1e-6
	error = int(args.error) * 1e-6
	wl = int(args.wl) * 1e-9
	zs = int(args.zs) * 1e-2
	

	#__________________________________________________________________
	# Mehrere Gitter / Wellenlängen Überlagerung
	
	'''
	sehr gute Ideen was wir machen können! Ich kommentiere die nur  erstmal aus,
	 sodass ich mit einer festen Wellenlänge und einem Gitter anfangen kann und nicht von den Fragen
	 genervt werde ;) Alle Parameter werden oben als Argument definiert, sodass ich die nicht immer 
	 eingeben muss. Siehe default value.

	i = int(input("Wie viele Wellenlängen sollen in der Welle überlagert sein? "))

	if i >= 1:
		counter = 1
		wl = list()
		while counter <= i:
			wltemp = float(input("Wellenlänge in nm? ")) # Lambda in nm
			wl.append(wl)
			counter += 1
	else: 
		wl = float(input("Wellenlänge in nm? ")) # Lambda in nm

	

	N = int(input("Anzahl der Gitter? "))
	alpha = float(input("Einfallswinkel? "))

	
	zs = 0
	while a*10 >= zs:
		zs = float(input("Abstand des Schirms in cm? "))
	'''

	#__________________________________________________________________
	# Schauen welche Funktion man ausführen muss spalt, gitter, gitterMitFehlstellen... 
	#comparefft(nx,ny,ax,ay,dx,dy,wl,zs)
	#spaltPeriodisch3d(nx,ny,ax,ay,dx,dy,wl,zs)
	#spaltAnyFunction3d(nx,ny,ax,ay,dx,dy,error,wl,zs)
	comparegriderrors(nx,ny,ax,ay,dx,dy,error,wl,zs) 	#vergleicht die Gitter/Beugungsmuster für Gitter mit und ohne Fehler
	
	
	#__________________________________________________________________
	# Ende der main()


####__________________________________________________________________
#### Hilfsvariablen/funktionen.
####__________________________________________________________________

def	k(wl):
	# Kreiswellenzahl
	return 2 * np.pi / wl 
def w(wl):
	# Kreisfrequenz Omega
	return c() * k(wl) 
def f(wl):
	# Frequenz
	return c() / wl 
def c():
	return float(299792458) # m/s

def sinc(x):
	return sin(x)/x

def i():
	return complex(0,1)

def dirac(x,mu):
	#ich weiß noch nicht wie es am besten ist hier dran zu gehen. 
	#Das hier ist eine gängige Approximation für mu->infinity
	return (np.abs(mu)/((pi)**0.5)) * exp(-(x*mu)**2)

#sollte keinen Unterschied fuer uns machen, da das Ergebnis immer real ist (zumindest fuer n Spalte)
def complex_int(func, a, b, **kwargs):
	def real_func(x):
		return scipy.real(func(x))
	def imag_func(x):
		return scipy.imag(func(x))
	real_integral = integrate.quad(real_func, a, b, **kwargs)
	imag_integral = integrate.quad(imag_func, a, b, **kwargs)
	return (real_integral[0] + i()*imag_integral[0])


####__________________________________________________________________
#### Berechnungsfunktionen mittels Fouriertransformation 
####__________________________________________________________________ 

def fourierNspaltPeriodisch(xArray,yArray,nx,ny,ax,ay,dx,dy,wl,zs):  ##funktioniert
	#Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
	#beim Integrieren bzw bei der fft schief geht.
	subArrayX= []
	subArrayY= []
	
	for x in xArray:
		if nx==0:
			subArrayX.append(1)
		else:
			subArrayX.append((float(fourierNspaltPeriodischIntegrate(x,nx,ax,dx,wl,zs))))
	for y in yArray:
		if ny==0:
			subArrayY.append(1)
		else:
			subArrayY.append((float(fourierNspaltPeriodischIntegrate(y,ny,ay,dy,wl,zs))))
		
	XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
	Ztmp=XX*YY

	return Ztmp

	
def fourierNspaltAnyFunction(xArray,yArray,nx,ny,ax,ay,dx,dy,error,wl,zs): ##gibt 1D richtiges Ergebnis
	## bietet die Möglichkeit in 'fourierNspaltIntegrateWithWholeTransmissionFunction(x,nx,ax,dx,wl,zs)' eine
	## beliebige Funktion für das Gitter einzusetzen
	
	#Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
	#beim Integrieren bzw bei der fft schief geht.
	subArrayX= []
	subArrayY= []
	
	for x in xArray:
		if nx==0:
			subArrayX.append(1)
		else:
			subArrayX.append(float(fourierNspaltIntegrateAnyFunction(x,nx,ax,dx,error,wl,zs)))
	for y in yArray:
		if ny==0:
			subArrayY.append(1)
		else:
			subArrayY.append(float(fourierNspaltIntegrateAnyFunction(y,ny,ay,dy,error,wl,zs)))

	XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
	Ztmp=XX*YY
	
	return Ztmp
	
def fourierNspaltIntegrateAnyFunction(x,n,a,d,error,wl,zs):
	# Fouriertransformierte von Transmission_Gitter
	
	## bietet die Möglichkeit eine beliebige Funktion für das Gitter in 'Transmission_n_Spalte(y,n,a,d)' einzusetzen
	
	u = k(wl)*sin(arctan(x/zs))
	#lambda x sagt python nur dass das die Variable ist und nach der integriert werden muss
	f = lambda y:  Transmission_n_Spalte(y,n,a,d,error)*exp(-i()*u*y) 

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
	#print(mittelpunkteDerLoecher)
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
	if math.fabs(x) < a/2:
		return 1
	else:
		return 0

def Transmission_Lochblende(rho,R):
	#einzelnes Loch mit Radius R
	#Verwende Polarkoordinaten rho,theta 
	if (rho < R):
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

def Transmission_n_Spalte(x,n,a,d,error):
	#error moves the slits(except the 0 order slit) by error further away from (0,0)
		
	gesamttransmission = 0.
	i = 1

	if (n % 2) == 1:
		gesamttransmission = Transmission_Einzelspalt(x,a)

	while i<=n/2:
		if (n % 2) == 0:
			gesamttransmission += Transmission_Einzelspalt(x-error-d*(2*i-1)/2,a) + Transmission_Einzelspalt (x+error+d*(2*i-1)/2,a)
		else:
			gesamttransmission += Transmission_Einzelspalt(x-error-d*i,a) + Transmission_Einzelspalt(x+error+d*i,a)
		i =i+1


	return gesamttransmission

#def Transmission_n_Spalte(x,n,a,d):
	


def Transmission_Gitter(xArray,yArray,nx,ny,ax,ay,dx,dy,error):
	# Returns the transmission for a periodic grid as a matrix with 0/1
	# to plot it with the contourplot-fct
	# error is the error of the grid which is given to the Transmission-fct
	
	subArrayX=[]
	subArrayY=[]
	for x in xArray:
		if nx==0:
			subArrayX.append(1)
		else:
			subArrayX.append(Transmission_n_Spalte(x,nx,ax,dx,error))
	for y in yArray:
		if ny==0:
			subArrayY.append(1)
		else:
			subArrayY.append(Transmission_n_Spalte(y,ny,ay,dy,error))

	XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
	Ztmp=XX*YY
	
	return Ztmp


####__________________________________________________________________
#### Intensitätsverteilungen für verschiedene Objekte. Ich weiß nicht ob
#### wir das am Ende so machen können. Für einen Einzelspalt geht es
####__________________________________________________________________


def interferenz_einzelspalt_manuell(X,a,wl,zs):

	alphax = arctan(X/zs)
	return (((a*sinc(0.5*a*k(wl)*sin(alphax))))**2)

def interferenz_Nspalt_manuell(X,n,a,d,wl,zs):
	alphax = arctan(X/zs)
	#alphay = arctan(Y/zs)
	return ((n*sin(pi*n*d/wl*sin(alphax))/(sin(pi*d/wl*sin(alphax))) * a * sinc(pi*a/wl*sin(alphax)))**2)


####__________________________________________________________________
#### Hauptfunktionen für n Spalte, Gitter, Gitter mit Fehlstelle etc..
#### Aufzurufen aus der main()
####__________________________________________________________________




def spaltPeriodisch3d(nx,ny,ax,ay,dx,dy,wl,zs):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	
	x1  = np.arange(-3., 3., 0.005)
	y1  = np.arange(-3., 3., 0.005)

	X,Y = np.meshgrid(x1, y1)
	Z = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)

	h = plt.contour(X,Y,Z,levels = np.linspace(np.min(Z), np.max(Z), 100))
	plt.show()
	
def spaltAnyFunction3d(nx,ny,ax,ay,dx,dy,error,wl,zs):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	
	x1  = np.arange(-3., 3., 0.005)
	y1  = np.arange(-3., 3., 0.005)

	X,Y = np.meshgrid(x1, y1)
	Z = fourierNspaltAnyFunction(x1,y1,nx,ny,ax,ay,dx,dy,error,wl,zs)
	
	h = plt.contour(X,Y,Z,levels = np.linspace(np.min(Z), np.max(Z), 100))
	plt.show()

def comparegriderrors(nx,ny,ax,ay,dx,dy,error,wl,zs):
	# n  : Anzahl der Spalte
    # a  : Größe der Spalte
    # d  : Abstand (egal für Einzelspalt)
    
    x_Spalt = np.array(np.linspace(-(nx-1)/2*dx-2*ax,(nx-1)/2*dx+2*ax,1200))
    y_Spalt = np.array(np.linspace(-(ny-1)/2*dy-2*ay,(ny-1)/2*dy+2*ay,1200))
    
    X_mat_Spalt, Y_mat_Spalt = np.meshgrid(x_Spalt,y_Spalt)
    
    z1 = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,0)
    z2 = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,error)
    
    x1  = np.arange(-5., 5., 0.005)
    y1  = np.arange(-5., 5., 0.005)
    
    X,Y = np.meshgrid(x1, y1)

    z4 = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
    z5 = fourierNspaltAnyFunction(x1,y1,nx,ny,ax,ay,dx,dy,error,wl,zs)
    
    ## Farbstufen für das Bild
    levels_z4 = [0, z4.max()/3000, z4.max()/1000, z4.max()/300, z4.max()/100, z4.max()/30, z4.max()/10, z4.max()]
    cmap_lin = plt.cm.Reds
    cmap_nonlin_z4 = nlcmap(cmap_lin, levels_z4)
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    
    plt.subplot(2,2,1)
    f = plt.pcolor(X_mat_Spalt*1000000, Y_mat_Spalt*1000000,z1, cmap='gray')
    
    plt.subplot(2,2,2)
    g = plt.pcolor(X_mat_Spalt*1000000, Y_mat_Spalt*1000000,z2, cmap='gray')
    
    plt.subplot(2,2,3)
    h = plt.contourf(X,Y,z4,levels=levels_z4,cmap=cmap_nonlin_z4)
    plt.colorbar()
    #h = plt.contour(X,Y,z3,levels = np.linspace(np.min(z3), np.max(z3), 100))
        
    plt.subplot(2,2,4)
    l = plt.contourf(X,Y,z5,levels=levels_z4,cmap=cmap_nonlin_z4)
    plt.colorbar()
    #l = plt.contour(X,Y,z4,levels = np.linspace(np.min(z3), np.max(z3), 100))
    
    plt.show()
	
def comparefft(nx,ny,ax,ay,dx,dy,wl,zs):
	# n  : Anzahl der Spalte
    # a  : Größe der Spalte
    # d  : Abstand (egal für Einzelspalt)
    
    x1  = np.arange(-5., 5., 0.005)
    y1  = np.arange(-5., 5., 0.005)
    
    X,Y = np.meshgrid(x1, y1)

    #z1 = fourierNspaltAnyFunction(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
    z2 = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
    
    ##NEW
    #dp=100 # *4(even) *2(odd) datapoints per slit
    datapoints=6000
    d=max(dx,dy)
    n=max(nx,ny)
    a=max(ax,ay)
    N = int(np.around(datapoints+1)) #datapoints in the whole array
    ##  4nd/a | N
    ## finde noch eine Funktion, die N berechnet
    print(N)
    x_Spalt = np.array(np.linspace(-d*n,d*n,N))
    y_Spalt = np.array(np.linspace(-d*n,d*n,N))
    
    dt=(x_Spalt[1]-x_Spalt[0])
    
    fa=1.0/dt
    z1_Xf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs
    z1_Yf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs
    
    
    index_low =  np.argmax(z1_Xf>-5.0)
    index_high = np.argmax(z1_Xf>5.0)
    XX,YY = np.meshgrid(z1_Xf[index_low:index_high],z1_Yf[index_low:index_high])
    
    ## 1D Berechnung
    Transmission_X=[]
    for x in x_Spalt:
        Transmission_X.append(Transmission_n_Spalte(x,nx,ax,dx,error))
    z1Dy=fftshift(np.square(np.abs(fft(Transmission_X))*2/N))
    
    
    ## 2D Berechnung
    z1 = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,error)
    z1f = fftshift(np.square(np.abs(fft2(z1))*4/N/N))[index_low:index_high,index_low:index_high]
    
    
    ## Berechnung mit Formel
    z1normalX=[]
    z1normalY=[]
    for x in x1:
        z1normalX.append(interferenz_Nspalt_manuell(x,nx,ax,dx,wl,zs))
        #z1normalX.append(interferenz_einzelspalt_manuell(x,ax,wl,zs))
    for y in y1:
        #z1normalY.append(interferenz_einzelspalt_manuell(y,ay,wl,zs))
        z1normalY.append(interferenz_Nspalt_manuell(y,ny,ay,dy,wl,zs))
    z1nX,z1nY = np.meshgrid(z1normalX,z1normalY)
    z1normal = z1nX*z1nY
    
    ## Farbstufen für das Bild
    levels_z1 = [0, z1f.max()/3000, z1f.max()/1000, z1f.max()/300, z1f.max()/100, z1f.max()/30, z1f.max()/10, z1f.max()]
    cmap_lin = plt.cm.Reds
    cmap_nonlin_z1 = nlcmap(cmap_lin, levels_z1)
    
    levels_z2 = [0, z2.max()/3000, z2.max()/1000, z2.max()/300, z2.max()/100, z2.max()/30, z2.max()/10, z2.max()]
    cmap_lin = plt.cm.Reds
    cmap_nonlin_z2 = nlcmap(cmap_lin, levels_z2)
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    plt.subplot(1,3,1)
    #h = plt.plot(z1_Xf[index_low:index_high],z1Dy[index_low:index_high])
    h = plt.contourf(XX,YY,z1f,levels=levels_z1,cmap=cmap_nonlin_z1)
    plt.colorbar()
    
    plt.subplot(1,3,2)
    #f = plt.plot(x1,z1normalX)
    f = plt.contourf(X,Y,z1normal,levels=levels_z2,cmap=cmap_nonlin_z2)
    plt.colorbar()
    
    plt.subplot(1,3,3)
    g = plt.contourf(X,Y,z2,levels=levels_z2,cmap=cmap_nonlin_z2)
    plt.colorbar()
          
    plt.show()
	
if __name__ == "__main__":
	main()






