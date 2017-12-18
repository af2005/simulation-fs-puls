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


	print('------------------------------------------------------------------------------')

	#__________________________________________________________________
	# Variablen auf SI Einheiten bringen. 
	wl = args.wl * 1e-9
	zs = args.zs * 1e-2
	a  = args.a  * 1e-6
	n  = args.n
	d  = args.d  * 1e-3
	h  = args.h  * 1e-3

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
	spalt(n,a,d,h,wl,zs)


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


####__________________________________________________________________
#### Transmissionsfunktion verschiedener Objekte
####__________________________________________________________________

def Transmission_Einzelspalt(x,a):
	#Einzelspalt der Dicke a
	#x ist die Variable

	if (math.fabs(x) < a/2):
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


####__________________________________________________________________
#### Intensitätsverteilungen für verschiedene Objekte. Ich weiß nicht ob
#### wir das am Ende so machen können. Für einen Einzelspalt geht es
####__________________________________________________________________
	

def interferenz_einzelspalt(X,Y,a,wl,zs):

	alphax = tan(X/zs)
	alphay = tan(Y/zs)
	return (((sinc(0.5*a*k(wl)*sin(alphax))))**2)

def interferenz_doppelspalt(X,Y):
	n=2
	
		
####__________________________________________________________________
#### Hauptfunktionen für n Spalte, Gitter, Gitter mit Fehlstelle etc..
#### Aufzurufen aus der main()
####__________________________________________________________________




def spalt(n,a,d,h,wl,zs):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	# h  : Hoehe des Spaltes (überlicherweise unendlich)
	
	if (n==1):
		x_1 = np.linspace(-3, 3, 300)
		y_1 = np.linspace(-3, 3, 300)
		X,Y = np.meshgrid(x_1,y_1)
		Z = interferenz_einzelspalt(X,Y,a,wl,zs).T
		plt.figure()
		#auf einem anderen Colourmesh wie gray sieht man nur das erste Maximum.
		plt.pcolormesh(Y,X, Z,cmap=plt.get_cmap("pink"))
		#plt.gca().set_aspect("equal") # x- und y-Skala im gleichen Maßstaab
		plt.show()
	elif (n==2):
		print('')
		#Fouriertransformierte von Transmission_Einzelspalt
		#Siehe Glg 29 im Theory doc.pdf

def gitter(a,wl,zs):
	print('nothing here')	

def gitterMitFehler(a,wl,zs,fehlerarray):
	print('nothing here')	


if __name__ == "__main__":
	main()





