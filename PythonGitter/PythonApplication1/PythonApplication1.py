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
import random

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
	parser.add_argument('--errortype', dest='errortype', help='Gitterfehlertyp',default=1)
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
	print('   Gitterfehler:       ' + str(args.errortype))
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
	error = int(args.error)
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

    error_matrix = []
    for i in range(ny):
        error_row=[]
        for j in range(nx):
            error_row.append([[random.uniform(-0.1,0.1),random.uniform(0.9,1.1)],[random.uniform(-0.1,0.1),random.uniform(0.9,1.1)]])
            #error_row.append([[0.5*j,0.5*(j+1)],[0.5*i,0.5*(i+1)]])
        error_matrix.append(error_row)
    error_matrix = np.array(error_matrix)

    #for i in range(ny):
    #    for j in range(nx):
    #        print(error_array[i,j,0])
    #comparefft(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
    #spaltPeriodisch3d(nx,ny,ax,ay,dx,dy,wl,zs)
    #spaltAnyFunction3d(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
    comparegriderrors(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
    
    #__________________________________________________________________
    # Ende der main()


####__________________________________________________________________
#### Hilfsvariablen/funktionen. Muss leider so. Python ist etwas eigen 
#### mit seinen globalen Variable. Im Prinzip existieren sie nicht. 
#### Jetzt kann man überall darauf zugreifen mit z.B. c(). 
#### Die Wellenlänge müssen wir aber leider mitschleppen.
####__________________________________________________________________

def k(wl):
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

def interferenz_einzelspalt_manuell(X,a,wl,zs):

    alphax = arctan(X/zs)
    return np.square((a*sinc(0.5*a*k(wl)*sin(alphax))))

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
                elif x > ((i-1)-(nx-2)/2*dx) and x <= (i-(nx-2)/2*dx):
                    subArrayX.append(float(fourierNspaltIntegrateAnyFunction(x,nx,ax,dx,errortype,error_matrix[j,i,0,:],wl,zs)))
                else:
                    subArrayX.append(1)
            
            for y in yArray:
                if ny==0:
                    subArrayY.append(1)
                elif y > ((j-1)-(ny-2)/2*dy) and y <= (j-(ny-2)/2*dy):
                    subArrayY.append(float(fourierNspaltIntegrateAnyFunction(y,ny,ay,dy,errortype,error_matrix[j,i,1,:],wl,zs)))
                else:
                    subArrayY.append(1)
                    
                    
            XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
            Ztmp.append(XX*YY)
            subArrayX= []
            subArrayY= []
    
    Ztotal=Ztmp[0]
    #for k in range(1,len(Ztmp)):
    #    Ztotal*=Ztmp[k]
    return Ztotal


def fftNspalt2D_XYZ(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs): ##Ergebnisform stimmt, Skalierung noch nicht
    datapoints = 1199
    d = max(dx,dy)
    n = max(nx,ny)
    a = max(ax,ay)
    N = int(np.around(datapoints+1)) #datapoints in the whole array
    ##  4nd/a | N-1
    ## finde noch eine Funktion, die N berechnet
    x_Spalt = np.array(np.linspace(-dx*nx/2,dx*nx/2,N))
    y_Spalt = np.array(np.linspace(-dy*ny/2,dy*ny/2,N))
    
    dx = (x_Spalt[1]-x_Spalt[0]) #Sampling-Rate ist für x- und y-Richtung gleich (momemtan)
    fa = 1.0/dx #Nyquist-Frequenz
    ## Nächste Kommentarzeile noch zu verbessern!##
    Xf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs  #Datenpunkte der fft als k-Vektoren, zurückgerechnet in x-/y-Positionen auf dem Schirm via Gl. ????
    Yf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs
    
    index_low =  np.argmax(Xf>-5.0) #Beschränke den Plot auf -5m bis +5m auf dem Screen
    index_high = np.argmax(Xf>5.0)
    X_Schirm, Y_Schirm = np.meshgrid(Xf[index_low:index_high],Yf[index_low:index_high])
    
    ## 2D Berechnung
    z2D = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,errortype,error_matrix)
    print(np.count_nonzero(z2D))
    z2Df = fftshift(np.square(np.abs(fft2(z2D))*4/N/N))[index_low:index_high,index_low:index_high]
    return X_Schirm, Y_Schirm, z2Df, z2D

def fftNspalt1D_XZ(nx,ax,dx,errortype,error_array,wl,zs): ##gibt 1D richtiges Ergebnis
    datapoints = 6000
    N = int(np.around(datapoints+1)) #datapoints in the whole array
    ##  4nd/a | N-1
    ## finde noch eine Funktion, die N berechnet
    x_Spalt = np.array(np.linspace(-dx*nx/2,dx*nx/2,N))
    
    dx = (x_Spalt[1]-x_Spalt[0]) #Sampling-Rate ist für x- und y-Richtung gleich (momemtan)
    fa = 1.0/dx #Nyquist-Frequenz
    ## Nächste Kommentarzeile noch zu verbessern!##
    Xf = tan(arcsin(np.linspace(-fa/2,fa/2,N)*wl))*zs  #Datenpunkte der fft als k-Vektoren, zurückgerechnet in x-/y-Positionen auf dem Schirm via Gl. ????
        
    index_low = np.argmax(Xf>-5.0) #Beschränke den Plot auf -5m bis +5m auf dem Screen
    index_high = np.argmax(Xf>5.0)
    X_Schirm = Xf[index_low:index_high]
    
    ## 1D Berechnung
    Transmission_X = []
    for x in x_Spalt:
        Transmission_X.append(Transmission_n_Spalte(x,nx,ax,dx,errortype,error_array))
    z1Df = fftshift(np.square(np.abs(fft(Transmission_X))*2/N))
    
    return X_Schirm, z1Df
    
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
        
    if errortype==0:  ##kein Error
        gesamttransmission = 0.
        

        '''
        i = 1
        if (n % 2) == 1:
            gesamttransmission = Transmission_Einzelspalt(x,a)

        while i<=n/2:
            if (n % 2) == 0:
                gesamttransmission += Transmission_Einzelspalt(x-d*(2*i-1)/2,a) + Transmission_Einzelspalt (x+d*(2*i-1)/2,a)
            else:
                gesamttransmission += Transmission_Einzelspalt(x-d*i,a) + Transmission_Einzelspalt(x+d*i,a)
            i =i+1'''
        for i in range(n):
            gesamttransmission += Transmission_Einzelspalt(x+d*(i-n/2+0.5),a)
    
    elif errortype==1:
        
        gesamttransmission = 0.
        for i in range(n):
            gesamttransmission += Transmission_Einzelspalt(x+error_array[0]*a+d*(i-n/2+0.5),a*error_array[1])
            
    #else:
        #errortype 2
    
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
    NoSlit=False #is set to True if either nx or ny==1, so there is no slit in that direction and the intensities will have to be multiplied instead of summed up for each quadrant
    
    for i in range(nx):
        for j in range(ny):
            for x in xArray:
                if nx==0:
                    subArrayX.append(1)
                    NoSlit=True
                elif x > ((i-1-(nx-2)/2)*dx) and x <= ((i-(nx-2)/2)*dx):
                    subArrayX.append(Transmission_n_Spalte(x,nx,ax,dx,errortype,error_matrix[j,i,0,:]))
                else:
                    subArrayX.append(0)
            for y in yArray:
                if ny==0:
                    subArrayY.append(1)
                    NoSlit=True
                elif y > ((j-1-(ny-2))/2*dy) and y <= ((j-(ny-2))/2*dy):
                    subArrayY.append(Transmission_n_Spalte(y,ny,ay,dy,errortype,error_matrix[j,i,1,:]))
                else:
                    subArrayY.append(0)
                    
                    
            XX, YY = np.meshgrid(np.array(subArrayX),np.array(subArrayY))
            Ztmp.append(XX*YY)
            subArrayX= []
            subArrayY= []
    
    Ztotal=Ztmp[0]
    for k in range(1,len(Ztmp)):
        if NoSlit==False:
            Ztotal+=Ztmp[k]
        else:
            Ztotal*=Ztmp[k]
    
    return np.array(Ztotal)
            
####__________________________________________________________________
#### Intensitätsverteilungen für verschiedene Objekte. Ich weiß nicht, ob
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
    
def spaltAnyFunction3d(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
    # n  : Anzahl der Spalte
    # a  : Größe der Spalte
    # d  : Abstand (egal für Einzelspalt)
    
    x1  = np.arange(-3., 3., 0.005)
    y1  = np.arange(-3., 3., 0.005)

    X,Y = np.meshgrid(x1, y1)
    Z = fourierNspaltAnyFunction(x1,y1,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
    
    h = plt.contour(X,Y,Z,levels = np.linspace(np.min(Z), np.max(Z), 100))
    plt.show()

def comparegriderrors(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
    # n  : Anzahl der Spalte
    # a  : Größe der Spalte
    # d  : Abstand (egal für Einzelspalt)
    
    x_Spalt = np.array(np.linspace(-nx*dx/2,nx*dx/2,1200))
    y_Spalt = np.array(np.linspace(-ny*dy/2,ny*dy/2,1200))
    
    X_mat_Spalt, Y_mat_Spalt = np.meshgrid(x_Spalt,y_Spalt)
    
    z1 = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,0,error_matrix)
    print(np.count_nonzero(z1))
    z2 = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,errortype,error_matrix)
    print(np.count_nonzero(z2))
    
    x1  = np.arange(-5., 5., 0.005)
    y1  = np.arange(-5., 5., 0.005)
    
    X,Y = np.meshgrid(x1, y1)

    z4 = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
    XX, YY, z2Df, z2D = fftNspalt2D_XYZ(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
    
    ## Farbstufen für das Bild
    levels_z4 = [0, z4.max()/3000, z4.max()/1000, z4.max()/300, z4.max()/100, z4.max()/30, z4.max()/10, z4.max()]
    cmap_lin = plt.cm.Reds
    cmap_nonlin_z4 = nlcmap(cmap_lin, levels_z4)
    
    levels_z2Df = [0, z2Df.max()/3000, z2Df.max()/1000, z2Df.max()/300, z2Df.max()/100, z2Df.max()/30, z2Df.max()/10, z2Df.max()]
    cmap_lin = plt.cm.Reds
    cmap_nonlin_z2Df = nlcmap(cmap_lin, levels_z2Df)
    
    fig, ax = plt.subplots(nrows=2, ncols=2)
    
    plt.subplot(2,2,1)
    f = plt.pcolor(X_mat_Spalt*1000000, Y_mat_Spalt*1000000,z1, cmap='gray')
    
    plt.subplot(2,2,2)
    g = plt.pcolor(X_mat_Spalt*1000000, Y_mat_Spalt*1000000,z2D, cmap='gray')
    
    plt.subplot(2,2,3)
    h = plt.contourf(X,Y,z4,levels=levels_z4,cmap=cmap_nonlin_z4)
    plt.colorbar()
    #h = plt.contour(X,Y,z3,levels = np.linspace(np.min(z3), np.max(z3), 100))
        
    plt.subplot(2,2,4)
    l = plt.contourf(XX,YY,z2Df,levels=levels_z2Df,cmap=cmap_nonlin_z2Df)
    plt.colorbar()
    #l = plt.contour(X,Y,z4,levels = np.linspace(np.min(z3), np.max(z3), 100))
    
    plt.show()
    
def comparefft(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
    # n  : Anzahl der Spalte
    # a  : Größe der Spalte
    # d  : Abstand (egal für Einzelspalt)
    
    x1  = np.arange(-5., 5., 0.005)
    y1  = np.arange(-5., 5., 0.005)
    
    X,Y = np.meshgrid(x1, y1)

    #Berechnung dft
    #z1 = fourierNspaltAnyFunction(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
    z3 = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
    
    ## Berechnung mit Formel
    z2normalX=[]
    z2normalY=[]
    for x in x1:
        z2normalX.append(interferenz_Nspalt_manuell(x,nx,ax,dx,wl,zs))
    for y in y1:
        z2normalY.append(interferenz_Nspalt_manuell(y,ny,ay,dy,wl,zs))
    z2nX,z2nY = np.meshgrid(z2normalX,z2normalY)
    z2normal = z2nX*z2nY    
    
    #Berechnung fft 2D
    XX, YY, z2Df, z2D = fftNspalt2D_XYZ(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
    
    ##  Berechnung fft 1D
    
    x1Df, z1Df = fftNspalt1D_XZ(nx,ax,dx,errortype,error_array,wl,zs)
    
    ## Farbstufen für das Bild
    levels_z1 = [0, z2Df.max()/3000, z2Df.max()/1000, z2Df.max()/300, z2Df.max()/100, z2Df.max()/30, z2Df.max()/10, z2Df.max()]
    cmap_lin = plt.cm.Reds
    cmap_nonlin_z1 = nlcmap(cmap_lin, levels_z1)
    
    levels_z3 = [0, z3.max()/3000, z3.max()/1000, z3.max()/300, z3.max()/100, z3.max()/30, z3.max()/10, z3.max()]
    cmap_lin = plt.cm.Reds
    cmap_nonlin_z3 = nlcmap(cmap_lin, levels_z3)
    
    fig, ax = plt.subplots(nrows=1, ncols=3)
    
    print(z2Df.max())
    print(z3.max())
    print(z2Df.max()/z3.max())
    
    plt.subplot(1,3,1)
    #h = plt.plot(x1Df, z1Df)
    h = plt.contourf(XX,YY,z2Df,levels=levels_z1,cmap=cmap_nonlin_z1)
    plt.colorbar()
    
    plt.subplot(1,3,2)
    #f = plt.plot(x1,z1normalX)
    f = plt.contourf(X,Y,z2normal,levels=levels_z3,cmap=cmap_nonlin_z3)
    plt.colorbar()
    
    plt.subplot(1,3,3)
    g = plt.contourf(X,Y,z3,levels=levels_z3,cmap=cmap_nonlin_z3)
    plt.colorbar()
          
    plt.show()
	
if __name__ == "__main__":
	main()






