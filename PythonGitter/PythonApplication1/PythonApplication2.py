def interferenz_doppelspalt_manuell(X,Y,a,d,wl,zs):
	n=2

    alphax = arctan(X/zs)

    #alphay = arctan(Y/zs)
    u = k(wl)*math.sin(alphax)
    #Formel 8 folgend
	#psi = integrate.quad(Transmission_n_Spalte(x,n,a,d)*exp(-i() * ( k()*sin(alphax)*x + k()*sin(alphay)*y) ),)
    return((cos(u*d/2)*sin(a*u/2)/(a*u/2)**2)**2)

def interferenz_Nspalt_manuell(X,Y,a,wl,zs,N):
    alphax = arctan(X/zs)
    #alphay = arctan(Y/zs)
return ((N*sin(pi*N*d/wl*sin(alphax))/(sin(pi*a/wl*sin(alphax))) * a * sin(pi*a/wl*sin(alphax))/(pi*a/wl*sin(alphax)))**2)