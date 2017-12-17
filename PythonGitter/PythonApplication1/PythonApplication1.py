#!/usr/bin/env python
# -*- coding: utf-8; -*-
#
# Copyright (C) 2016 Bernd Lienau, Simon Jung, Alexander Franke

import math
import cmath

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

#wird automatisch aufgerufen
def main():
	arser = argparse.ArgumentParser(description='This is a python3 module simulating a light pulse with given parameters propagating through different optical components suchas Einzelspalt, Doppelspalt, Gitter mit und ohne Fehlstellen oder Defekten.')
    parser.add_argument('--dimension', dest='dimension',required=True,help='Auf 1 zu setzen für n Spalte, auf 2 für Gitter .',default=1)
    parser.add_argument('--spalte', dest='n_spalte', help='Die Anzahl der Spalte. Ganzzahlige Zahl zwischen 1 und Unendlich.',default=1)
    parser.add_argument('--gitterkonst', dest='a', help='Gitterkonstante/Spaltbreite in mm',default=2)
    parser.add_argument('--wellenlaenge', dest='wl',required=True,help='Wellenlänge in nm',default=800 )
    parser.add_argument('--schirmabstand', dest='zs', required=True, help='Schirmabstand in cm',default=50)
    args = parser.parse_args()

   
	print('----------------------------------------------------------------------------------')
    print('Die Kunst der Computer-basierten Modellierung und Simulation experimenteller Daten')
    print('----------------------------------------------------------------------------------')
    print('')
    print('                          Projekt Lichtpuls Simulation                            ')
    print('')
    print('                von Bernd Lienau, Simon Jung, Alexander Franke                    ')
    print('')
    print('----------------------------------------------------------------------------------')
    print('')
    print('Es wurden folgende Parameter eingegeben oder angenommen'                           )
    print('   Dimension                                  :  ' + args.dimension                )
    print('   Falls Dimension = 1, berechnen wir für     :  ' + args.spalte + ' Spalte'       )
    print("   Falls Dimesnion = 2 ,Gitterkonstante in mm :  " + args.gitterkonst              )
    print("   Wellenlänge in nm 					     :  " + args.wl                       )


    print('----------------------------------------------------------------------------------')

    #__________________________________________________________________
	# Variablen auf SI Einheiten bringen. 
	
	wl       = wl * 1e9
	zs       = zs * 1e-2
	a        = a  * 1e-3

	#__________________________________________________________________
	# Mehrere Gitter / Wellenlängen Überlagerung
	
	'''
	sehr gute Ideen was wir machen können! Ich kommentiere die nur  erstmal aus,
	 sodass ich mit einer festen Wellenlänge und einem Gitter anfangen kann und nicht von den Fragen
	 genervt werde ;) Einige Parameter werden oben als Argument definiert, sodass ich die nicht immer 
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
	# Schauen welche Funktion man ausführen muss 
	spalt()


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
	return 2 * math.pi / wl 
def w(wl):
	# Kreisfrequenz Omega
	return = c() * k(wl) 
def f(wl):
	# Frequenz
	return c() / wl 
def c():
	return float(299792458) # m/s


####__________________________________________________________________
#### Transmissionsfunktion verschiedener Objekte
####__________________________________________________________________

def Transmission_Einzelspalt(x,d):
	#Einzelspalt der Dicke d
	#x ist die Variable

	if (math.abs(x) < d/2):
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
		


#__________________________________________________________________
# n  : Anzahl der Spalte
# d  : Abstand (egal für Einzelspalt)
# wl : Wellenlänge
# 
def spalt(n,d,wl,sz):

	#__________________________________________________________________
	# Definition der Funktion

	f(x)



	#__________________________________________________________________
	# Teil für das Ausrechnen der Intensität am Ort x des Schirms
	# später mit Fehlstellen
	# Erweiterung auf 2. Dimension




if __name__ == "__main__":
    main()





