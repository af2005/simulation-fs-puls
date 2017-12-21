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


import csv
import pandas as pd
import scipy
from scipy import integrate as integrate


import matplotlib.pyplot as plt
import argparse

#main() wird automatisch aufgerufen
def main():
	parser = argparse.ArgumentParser(description='This is a python3 module simulating a light pulse with given parameters propagating through different optical components suchas Einzelspalt, Doppelspalt, Gitter mit und ohne Fehlstellen oder Defekten.')
	parser.add_argument('--dimension', dest='dimension',help='Auf 1 zu setzen für n Spalte, auf 2 für Gitter .',default=2)
	parser.add_argument('--Spalte in x-Richtung', dest='nx', help='Die Anzahl der Spalte in x-Richtung. Ganzzahlige Zahl zwischen 1 und Unendlich.',default=1)
	parser.add_argument('--Spalte in y-Richtung', dest='nx', help='Die Anzahl der Spalte in x-Richtung. Ganzzahlige Zahl zwischen 1 und Unendlich.',default=1)
	parser.add_argument('--Spaltbreite in x-Richtung', dest='ax', help='Spaltbreite in um',default=3)
	parser.add_argument('--Spaltbreite in y-Richtung', dest='ay', help='Spalthoehe in um',default=5)
	parser.add_argument('--Spaltabstand in x-Richtung', dest='dx', help='Spaltabstand in horizontaler Richtung in um',default=10)
	parser.add_argument('--Spaltabstand in y-Richtung', dest='dy', help='Spaltabstand in vertikaler Richtung in um',default=10)
	parser.add_argument('--wellenlaenge', dest='wl',help='Wellenlänge in nm',default=780 )
	parser.add_argument('--schirmabstand', dest='zs', help='Schirmabstand in cm',default=350)
	


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
	print('   Dimension                                  :  ' + str(args.dimension))
	print('   Falls Dimension = 1, berechnen wir für     :')
	print('      ' + str(args.nx) + ' Spalte in horizontaler Richtung(x)')
	print('      ' + str(args.ny) + ' Spalte in vertikaler Richtung(y)')
	print('   Spaltbreite in um          :')
	print('      horizontal(x):   ' + str(args.ax))
	print('      vertikal(y):     ' + str(args.ay))
	print('   Spaltabstand in um          :')
	print('      horizontal(x):   ' + str(args.dx))
	print('      vertikal(y):     ' + str(args.dy))
	print("   Wellenlänge in  nm                         :  " + str(args.wl))
	print("   Schirmabstand in cm                        :  " + str(args.zs))
	print('')
	print('------------------------------------------------------------------------------')

	#__________________________________________________________________
	# Variablen auf SI Einheiten bringen. 
	nx = int(args.nx)
	ny = int(args.ny)
	ax = args.ax  * 1e-6
	ay = args.ay  * 1e-6
	dx  = args.dx  * 1e-6
	dy  = args.dy  * 1e-6
	wl = args.wl * 1e-9
	zs = args.zs * 1e-2
	

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
	spalt3d(n,a,d,h,wl,zs)


	#__________________________________________________________________
	# Ende der main()


####__________________________________________________________________
#### Hilfsvariablen/funktionen. Muss leider so. Python ist etwas eigen 
#### mit seinen globalen Variable. Im Prinzip existieren sie nicht. 
#### Jetzt kann man überall darauf zugreifen mit z.B. c(). 
#### Die Wellenlänge müssen wir aber leider mitschleppen.
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

def fourierNspalt(xArray,yArray,nx,ny,ax,ay,dx,dy,wl,zs):  ##funktioniert
    #Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
    #beim Integrieren bzw bei der fft schief geht.
    subArrayX= []
    subArrayY= []
    
    for x in xArray:
        subArrayX.append((float(fourierNspaltIntegrate(x,nx,ax,dx,wl,zs))))
    for y in yArray:
        subArrayY.append((float(fourierNspaltIntegrate(y,ny,ay,dy,wl,zs))))
        
    XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
    Ztmp=XX*YY

    return Ztmp

## neue Funktionen für die Integration über beliebige Gitterformen
## (still to be tested, work for normal grids)
## TBD: change discrete fourier transform to fft
	
def fourierNspaltWithWholeTransmissionFunction(xArray,yArray,nx,ny,ax,ay,dx,dy,wl,zs): ##gibt 1D richtiges Ergebnis
    ## bietet die Möglichkeit in 'fourierNspaltIntegrateWithWholeTransmissionFunction(x,nx,ax,dx,wl,zs)' eine
    ## beliebige Funktion für das Gitter einzusetzen
    
    #Diese Funktion dient nur dafuer nicht mit einem Array an x Werten arbeiten zu muessen, was 
    #beim Integrieren bzw bei der fft schief geht.
    subArrayX= []
    subArrayY= []
    
    for x in xArray:
        subArrayX.append(float(fourierNspaltIntegrateWithWholeTransmissionFunction(x,nx,ax,dx,wl,zs)))
    for y in yArray:
        subArrayY.append(float(fourierNspaltIntegrateWithWholeTransmissionFunction(y,ny,ay,dy,wl,zs)))

    XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
    Ztmp=XX*YY
    
    return Ztmp
	
def fourierNspaltIntegrateWithWholeTransmissionFunction(x,n,a,d,wl,zs):
    # Fouriertransformierte von Transmission_Gitter
    
    ## bietet die Möglichkeit eine beliebige Funktion für das Gitter in 'Transmission_n_Spalte(y,n,a,d)' einzusetzen
    
    u = k(wl)*sin(arctan(x/zs))
    #lambda x sagt python nur dass das die Variable ist und nach der integriert werden muss
    f = lambda y:  Transmission_n_Spalte(y,n,a,d)*exp(-i()*u*y) 

    integral = complex_int(f,-(n-1)*d/2-a,(n-1)*d/2+a)
    #scipy.real koennte man weg lassen, da korrekterweise der imaginaer Teil immer null ist. Aber damit
    #matplot keine Warnung ausgibt, schmeissen wir den img Teil hier weg.
    integral =  scipy.real(np.square(np.multiply(n,integral)))
    return integral
	
## Ende der beiden Funktionen für die Integration über beliebige Gitterformen
	
def fourierNspaltIntegrate(x,n,a,d,wl,zs):
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

def Transmission_n_Spalte(x,n,a,d):

    gesamttransmission = 0.
    i = 1

    if (n % 2) == 1:
        gesamttransmission = Transmission_Einzelspalt(x,a)

    while i<=n/2:
        if (n % 2) == 0:
            gesamttransmission += Transmission_Einzelspalt(x-d*(2*i-1)/2,a) + Transmission_Einzelspalt (x+d*(2*i-1)/2,a)
        else:
            gesamttransmission += Transmission_Einzelspalt(x-d*i,a) + Transmission_Einzelspalt(x+d*i,a)
        i =i+1


    return gesamttransmission

def Transmission_Gitter(x,y,n,a,d):
    return Transmission_n_Spalte(x,n,a,d) * Transmission_n_Spalte(y,n,a,d)

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

'''	
#vermutlich Falsch! Max Intensitaet
def interferenz_doppelspalt_manuell2(X,a,d,wl,zs): 
	n=2
	alphax = arctan(X/zs)
	#alphay = arctan(Y/zs)
	u = k(wl)*sin(alphax)
	#Formel 8 folgend
	#psi = integrate.quad(Transmission_n_Spalte(x,n,a,d)*exp(-i() * ( k()*sin(alphax)*x + k()*sin(alphay)*y) ),)
	return((cos(u*d/2)*sin(a*u/2)/(a*u/2))**2)	
'''
####__________________________________________________________________
#### Hauptfunktionen für n Spalte, Gitter, Gitter mit Fehlstelle etc..
#### Aufzurufen aus der main()
####__________________________________________________________________




def spalt3d(nx,ny,ax,ay,dx,dy,wl,zs):
    # n  : Anzahl der Spalte
    # a  : Größe der Spalte
    # d  : Abstand (egal für Einzelspalt)
    
    x1  = np.arange(-3., 3., 0.005)
    y1  = np.arange(-3., 3., 0.005)

    X,Y = np.meshgrid(x1, y1)
    Z = fourierNspaltWithWholeTransmissionFunction(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
    #Z = fourierNspalt(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)

    h = plt.contour(X,Y,Z,levels = np.linspace(np.min(Z), np.max(Z), 100))
    plt.show()

'''
if (n==1):
		x_1 = np.linspace(-3, 3, 300)
		y_1 = np.linspace(-3, 3, 300)
		X,Y = np.meshgrid(x_1,y_1)
		Z = interferenz_einzelspalt(X,Y,a,wl,zs).T
		A = interferenz_einzelspalt_fft_1d(X,Y,a,wl,zs).T #Mit fft erstellt
		fig, ax = plt.subplots(nrows=2, ncols=1)
		#auf einem anderen Colourmesh wie gray sieht man nur das erste Maximum.
		plt.subplot(2, 1, 1)
		plt.pcolormesh(Y,X, Z,cmap=plt.get_cmap("pink"))
		#plt.gca().set_aspect("equal") # x- und y-Skala im gleichen Maßstaab
		plt.subplot(2, 1, 2)
		plt.pcolormesh(Y,X, A,cmap=plt.get_cmap("green"))
		plt.show()
	elif (n==2):
'''

'''
def spalt2d(n,a,d,h,wl,zs):

	## sollte durch die auf 3D veränderten Funktionen nicht mehr gehen.
	## Sonst muss man die Returnvalues zurück auf subArrayX/subArrayY ändern.


	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	# h  : Hoehe des Spaltes (überlicherweise unendlich)
	t1 = np.arange(-3., 3., 0.005)
	t2 = t1
	plt.figure(1)
	plt.subplot(211)
	#plt.plot(t1,fourierEinzelspalt(arcsin(t1/zs),a,wl,lowerrange,upperrange) , 'r--')
	#(xArray,a,wl,n,d,zs)
	plt.plot(t1,fourierNspalt(t1,a,wl,n,d,zs) , 'r-')
	plt.subplot(212)
	plt.plot(t2,interferenz_Nspalt_manuell(t2,a,d,wl,zs,n),'b.')
	plt.show()
'''	

def gitter(a,wl,zs):
	print('nothing here')	

def gitterMitFehler(a,wl,zs,fehlerarray):
	print('nothing here')	


if __name__ == "__main__":
	main()





