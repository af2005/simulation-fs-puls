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
import scipy.integrate as integrate

import matplotlib.pyplot as plt
import argparse

#main() wird automatisch aufgerufen
def main():
	parser = argparse.ArgumentParser(description='This is a python3 module simulating a light pulse with given parameters propagating through different optical components suchas Einzelspalt, Doppelspalt, Gitter mit und ohne Fehlstellen oder Defekten.')
	parser.add_argument('--dimension', dest='dimension',help='Auf 1 zu setzen für n Spalte, auf 2 für Gitter .',default=1)
	parser.add_argument('--spalte', dest='n', help='Die Anzahl der Spalte. Ganzzahlige Zahl zwischen 1 und Unendlich.',default=1)
	parser.add_argument('--gitterkonst', dest='a', help='Gitterkonstante/Spaltbreite in um',default=3)
	parser.add_argument('--wellenlaenge', dest='wl',help='Wellenlänge in nm',default=800 )
	parser.add_argument('--schirmabstand', dest='zs', help='Schirmabstand in cm',default=350)
	parser.add_argument('--spaltbreite', dest='d', help='Spaltbreite in mm',default=1)
	parser.add_argument('--spalthoehe', dest='h', help='Spalthoehe in mm',default=20)


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
	print('   Falls Dimension = 1, berechnen wir für     :  ' + str(args.n) + ' Spalte')
	print("   Gitterkonstante/Spaltbreite in um          :  " + str(args.a))
	print("   Wellenlänge in  nm                         :  " + str(args.wl))
	print("   Schirmabstand in cm                        :  " + str(args.zs))
	print('')
	print('------------------------------------------------------------------------------')

	#__________________________________________________________________
	# Variablen auf SI Einheiten bringen. 
	wl = args.wl * 1e-9
	zs = args.zs * 1e-2
	a  = args.a  * 1e-6
	n  = args.n
	d  = args.d  * 1e-3
	h  = args.h  * 1e-3

	lowerrange = -n*d-a/2
	upperrange = n*d+a/2

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
	spalt(n,a,d,h,wl,zs,lowerrange,upperrange)


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
	 
def fourierEinzelspalt(xArray,a,wl,lowerrange,upperrange,d):
	output = []
	for value in xArray:
		output.append(fourierEinzelspaltIntegrate(value,a,wl,lowerrange,upperrange,d))
	return output

def fourierEinzelspaltIntegrate(alphax,a,wl,lowerrange,upperrange,d):
	u = k(wl)*math.sin(alphax)
	#print(u)
	f = lambda x: Transmission_n_Spalte(x,2,a,d) *exp(-i()*u*x)
	r = 1#(1/(2*pi)**0.5)
	integral =  np.square(np.multiply(integrate.quad(f,lowerrange,upperrange),r))
		
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
	if (rho < R):
		return 1
	else: 
		return 0

def Transmission_n_Spalte(x,n,a,d):
	# Fouriertransformierte von Transmission_Einzelspalt
	# Siehe Glg 29 im Theory doc.pdf
	# https://en.wikipedia.org/wiki/Dirac_delta_function#Translation folgend
	# ist f(t) gefaltet mit dirac(t-T) ist gleich f(t-T)
	# außerdem gilt distributivität (a+b) (*) c = a(*)c + b(*)c
	# für den Doppelspalt bzw. n-Spalt haben wir also 
	gesamttransmission = 0
	i = 1

	if n==1:
		gesamttransmission = Transmission_Einzelspalt(x,a)
		
	while i<=n/2:
		if n%2==0:
			gesamttransmission += Transmission_Einzelspalt(x-d*(2*i-1)/2,a)
			gesamttransmission += Transmission_Einzelspalt(x+d*(2*i-1)/2,a)
		else:
			gesamttransmission += Transmission_Einzelspalt(x-d*i,a)
			gesamttransmission += Transmission_Einzelspalt(x+d*i,a)
		i =i+1
	
	
	
	return gesamttransmission

def Transmission_Gitter(x,y,n,a,d):
	return Transmission_n_Spalte(x,n,a,d) * Transmission_n_Spalte(y,n,a,d)

####__________________________________________________________________
#### Intensitätsverteilungen für verschiedene Objekte. Ich weiß nicht ob
#### wir das am Ende so machen können. Für einen Einzelspalt geht es
####__________________________________________________________________
	

def interferenz_einzelspalt_manuell(X,Y,a,wl,zs):

	alphax = arctan(X/zs)
	alphay = arctan(Y/zs)
	return (((a*sinc(0.5*a*k(wl)*sin(alphax))))**2)

def interferenz_doppelspalt_manuell(X,Y,a,d,wl,zs):
	n=2
	alphax = arctan(X/zs)
	alphay = arctan(Y/zs)
	#Formel 8 folgend
	#psi = integrate.quad(Transmission_n_Spalte(x,n,a,d)*exp(-i() * ( k()*sin(alphax)*x + k()*sin(alphay)*y) ),)


	
		
####__________________________________________________________________
#### Hauptfunktionen für n Spalte, Gitter, Gitter mit Fehlstelle etc..
#### Aufzurufen aus der main()
####__________________________________________________________________




def spalt(n,a,d,h,wl,zs,lowerrange,upperrange):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	# h  : Hoehe des Spaltes (überlicherweise unendlich)
	if (n==1):
		t1 = np.arange(-3., 3., 0.1)
		t2 = t1
		plt.figure(1)
		plt.subplot(211)
		#plt.plot(t1,fourierEinzelspalt(arcsin(t1/zs),a,wl,lowerrange,upperrange,d) , 'r--')
		
		##change
		arrayX=[]
		for intx in np.arange(-0.0015, 0.0015, 0.000005):
			arrayX.append(Transmission_n_Spalte(intx,2,a,d))
		plt.plot(np.arange(-0.0015, 0.0015, 0.000005),np.array(arrayX), 'r--')
		plt.xlim(-0.05, 0.05)
		plt.ylim(-1,2)
		##change end
		
		plt.subplot(212)
		plt.plot(t2,interferenz_einzelspalt_manuell(t2,0,a,wl,zs),'b--')
		plt.show()
	elif (n==2):
		print('')

		

def gitter(a,wl,zs):
	print('nothing here')	

def gitterMitFehler(a,wl,zs,fehlerarray):
	print('nothing here')	


if __name__ == "__main__":
	main()





