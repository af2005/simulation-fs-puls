#!/usr/bin/env python
# -*- coding: utf-8; -*-
#
# Copyright (C) 2017 Bernd Lienau, Simon Jung, Alexander Franke

import math
import cmath
import numpy as np
import sys
import time

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
import matplotlib.gridspec as gridspec
from pylab import *
 

import csv
import pandas as pd
import scipy
from scipy import integrate as integrate
import random


import matplotlib.pyplot as plt
import argparse

#Canvas
from tkinter import *

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
	parser.add_argument('--errortype', dest='errortype', help='Gitterfehlertyp',default=0)
	parser.add_argument('--wl', dest='wl',help='Wellenlänge in nm',default=780 )
	parser.add_argument('--abstand', dest='zs', help='Schirmabstand in cm',default=350)
	parser.add_argument('--custom', dest='custom',help='Setze auf 1 um eine Leinwand zu haben und ein Beugungsmuster eines beliebigen Objekts zu bekommen',default=0 )
	


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
	errortype = int(args.errortype)
	wl = int(args.wl) * 1e-9
	zs = int(args.zs) * 1e-2
	custom = int(args.custom)
	
	
	matplotlib.rcParams.update({'font.size': 30}) ## change font size

	
	if nx==0 and ny==0:
		print('Ohne Gitter gibt es keine Berechnung...')
		sys.exit(0)


	if custom == 1:
		#baue canvas

		canvas_size = 200
		drawradius = 20

		imagearray =  [[ 0 for xcoord in range( canvas_size ) ] for ycoord in range( canvas_size ) ]



		def getNeightbourPixels(x0,y0):
			#gets all neightbouring pixels within a certain distance
			x0 = int(x0)
			y0 = int(y0)
			newdrawradius = drawradius + 1
			def abstand(x0,y0,x,y):
				return math.trunc(math.sqrt((x-x0)**2 + (y-y0)**2))
			tempx = x0-newdrawradius
			tempy = y0-newdrawradius

			if tempx < 0 :
				tempx = 0
			if tempy < 0:
				tempy = 0

			while tempx < canvas_size:
				tempy = y0-newdrawradius
				while (tempy < y0+newdrawradius) and tempy < canvas_size:
					if (abstand(x0,y0,tempx,tempy) <= newdrawradius):
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

		master = Tk()
		master.title( "Beugungsmuster" )
		w = Canvas(master, 
				   width=canvas_size, 
				   height=canvas_size,
				   bg="#000000")
		w.pack(expand = NO, fill = BOTH)
		w.bind("<B1-Motion>", paint)

		message = Label( master, text = "Press and Drag the mouse to draw" )
		message.pack( side = BOTTOM )
			
		mainloop()
		
		for row in imagearray:
			rowcontent = ""
			for entry in row:
				rowcontent += str(entry)
			print(rowcontent)
		X_trans, Y_trans = np.meshgrid(np.linspace(-trans.shape[1]/2*1e-7,trans.shape[1]/2*1e-7,trans.shape[1]), np.linspace(-trans.shape[0]/2*1e-7,trans.shape[0]/2*1e-7,trans.shape[0]))
		trans=np.array(imagearray)
		X,Y,Z = fftCanvas2D_XYZ(np.array(imagearray),wl,zs)
		Z /= np.nanmax(Z)
		
		levels_Z = [0, 1./1000., 1./300., 1./100., 1./30., 1./10., 1./3., 1.]
		cmap_lin = plt.cm.Reds
		cmap_nonlin_Z = nlcmap(cmap_lin, levels_Z)
		
		fig, ax = plt.subplots(nrows=1, ncols=2)
	
		plt.subplot(1,2,1)
		plt.contourf(X_trans,Y_trans,trans,cmap='gray')
		
		plt.subplot(1,2,2)
		plt.contourf(X,Y,Z,levels=levels_Z,cmap=cmap_nonlin_Z)
		plt.colorbar()
				
		plt.show()
	else:
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
				error_row.append([[random.uniform(-0.2,0.2),random.uniform(0.9,1.1)],[random.uniform(-0.2,0.2),random.uniform(0.9,1.1)]])
			error_matrix.append(error_row)
		error_matrix = np.array(error_matrix)

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

def kgV2(a, b):
	## gibt das KgV der beiden ints a und b zurück
	return (a * b) // math.gcd(a, b)

def kgV_arr(numbers):
	## gibt das kgV einer beliebigen Liste von Ints zurück
	kgV = numbers[0]
	for i in range(1,len(numbers)):
		kgV=kgV2(kgV,numbers[i])
	return kgV

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
		while(datapoints*wl/4/n/d/10<0.82):                 ## erhöhe Datapoints, damit mindestens die Raumfrequenzen berechnet werden, die auf dem Schirm abgebildet werden
			datapoints*=2                                   ## 0.82 für Plot bis +-5m
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
	while(datapoints*wl/4/nx/dx/10<0.82):                 ## erhöhe Datapoints, damit mindestens die Raumfrequenzen berechnet werden, die auf dem Schirm abgebildet werden
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
		
	if errortype==0:  ##kein Error
		gesamttransmission = 0.
		
		for i in range(n):
			gesamttransmission += Transmission_Einzelspalt(x-d*(i-n/2+0.5),a)
	
	elif errortype==1:
		
		gesamttransmission = 0.
		for i in range(n):
			gesamttransmission += Transmission_Einzelspalt(x+error_array[i,0]*a-d*(i-n/2+0.5),a*error_array[i,1])
			
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




def spaltPeriodisch3d(nx,ny,ax,ay,dx,dy,wl,zs):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	
	x1  = np.linspace(-5., 5., 1200)
	y1  = np.linspace(-5., 5., 1200)

	X,Y = np.meshgrid(x1, y1)
	Z = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)

	h = plt.contour(X,Y,Z,levels = np.linspace(np.min(Z), np.max(Z), 100))
	plt.show()
	
def spaltAnyFunction3d(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	
	x1  = np.linspace(-5., 5., 1200)
	y1  = np.linspace(-5., 5., 1200)

	X,Y = np.meshgrid(x1, y1)
	Z = fourierNspaltAnyFunction(x1,y1,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
	
	h = plt.contour(X,Y,Z,levels = np.linspace(np.min(Z), np.max(Z), 100))
	plt.show()

def comparegriderrors(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	start_time = time.time()
	
	d=max(dx,dy)
	n=max(nx,ny)
	a=max(ax,ay)
	x_Spalt = np.array(np.linspace(-n*d/2,n*d/2,1200))
	y_Spalt = np.array(np.linspace(-n*d/2,n*d/2,1200))
	
	X_mat_Spalt, Y_mat_Spalt = np.meshgrid(x_Spalt,y_Spalt)
	
	z1 = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,0,error_matrix)
	z2 = Transmission_Gitter(x_Spalt,y_Spalt,nx,ny,ax,ay,dx,dy,errortype,error_matrix)
	
	x1  = np.linspace(-5., 5., 1200)
	y1  = np.linspace(-5., 5., 1200)
	
	X,Y = np.meshgrid(x1, y1)

	z4 = interferenz_Gitter_analytisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
	z4 /= np.nanmax(z4) #normalization
	print("Analytische Berechnungen dauerten: " + str(time.time() - start_time))

	start_time = time.time()

	XX, YY, z2Df = fftNspalt2D_XYZ(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
	z2Df /= np.nanmax(z2Df)
	
	print("FFT Berechnungen dauerten: " + str(time.time() - start_time))
	start_time = time.time()

	## Farbstufen für das Bild
	levels_z4 = [0, 1./1000., 1./300., 1./100., 1./30., 1./10., 1./3., 1.]
	cmap_lin = plt.cm.Reds
	cmap_nonlin_z4 = nlcmap(cmap_lin, levels_z4)

	#fig, ax = plt.subplots(nrows=2, ncols=2)
	gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3]) 
	plt.subplot(gs[0, 0])
	f = plt.pcolor(X_mat_Spalt*1000000, Y_mat_Spalt*1000000,z1, cmap='gray')
	
	plt.subplot(gs[0, 1])
	g = plt.pcolor(X_mat_Spalt*1000000, Y_mat_Spalt*1000000,z2, cmap='gray')
	
	plt.subplot(gs[1, 0])
	h = plt.contourf(X,Y,z4,levels=levels_z4,cmap=cmap_nonlin_z4)
	plt.subplot(gs[1, 0]).set_title("analytisch")
	plt.colorbar()
			
	plt.subplot(gs[1, 1])
	l = plt.contourf(XX,YY,z2Df,levels=levels_z4,cmap=cmap_nonlin_z4)
	plt.subplot(gs[1, 1]).set_title("fft")
	plt.colorbar()
		
	plt.show()
	print("Plot Berechnungen dauerten: " + str(time.time() - start_time))
	
def comparefft(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs):
	# n  : Anzahl der Spalte
	# a  : Größe der Spalte
	# d  : Abstand (egal für Einzelspalt)
	start_time = time.time()
	
	x1  = np.linspace(-5., 5., 1200)
	y1  = np.linspace(-5., 5., 1200)
	
	X,Y = np.meshgrid(x1, y1)

	#Berechnung dft
	z1 = fourierNspaltPeriodisch(x1,y1,nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
	z1 /= z1.max()
	
	## Berechnung analytisch
	z2 = interferenz_Nspalt_analytisch(x1,y1,nx,ny,ax,ay,dx,dy,wl,zs)
	z2 /= z2.max()
	print("Analytische Berechnungen dauerten: " + str(time.time() - start_time))
	start_time = time.time()
	
	#Berechnung fft 2D
	XX, YY, z2Df = fftNspalt2D_XYZ(nx,ny,ax,ay,dx,dy,errortype,error_matrix,wl,zs)
	z2Df/=z2Df.max()
	
	print("FFT Berechnungen dauerten: " + str(time.time() - start_time))
	start_time = time.time()
	
	## Farbstufen für das Bild
	levels_z1 = [0, 1./1000., 1./300., 1./100., 1./30., 1./10., 1./3., 1.]
	cmap_lin = plt.cm.Reds
	cmap_nonlin_z1 = nlcmap(cmap_lin, levels_z1)
	
	fig, ax = plt.subplots(nrows=1, ncols=3)
	
	plt.subplot(1,3,1)
	g = plt.contourf(x1,y1,z1,levels=levels_z1,cmap=cmap_nonlin_z1)
	plt.subplot(1,3,1).set_title("dft ohne Gitterfehler")
	plt.colorbar()
	
	plt.subplot(1,3,2)
	f = plt.contourf(x1,y1,z2,levels=levels_z1,cmap=cmap_nonlin_z1)
	plt.subplot(1,3,2).set_title("analytisch")
	plt.colorbar()
	
	plt.subplot(1,3,3)    
	h = plt.contourf(XX,YY,z2Df,levels=levels_z1,cmap=cmap_nonlin_z1)
	plt.subplot(1,3,3).set_title("fft")
	plt.colorbar()

	print("Plot Berechung dauerte: " + str(time.time() - start_time))
		  
	plt.show()
	
	
if __name__ == "__main__":
	main()






