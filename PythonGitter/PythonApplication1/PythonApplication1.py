
import math
import cmath

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

c = float(299792458) # m/s
print(c)

a = float(input("Gittergröße in mm? ")) # Gitteröffnung in mm
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

k = 2 * math.pi / wl # Wellenzahl
w = c * k # Kreisfrequenz Omega
f = c / wl # Frequenz



#__________________________________________________________________
# Definition der Funktion




#__________________________________________________________________
# Teil für das Ausrechnen der Intensität am Ort x des Schirms
# später mit Fehlstellen
# Erweiterung auf 2. Dimension










