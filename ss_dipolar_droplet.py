###############################################
######### split-step dipolar doplet ###########
############ Matthew Edmonds 2019 #############
###############################################

from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np
import time
#
def integrand(t):
	return np.exp(-t) / t
#
def expint(u):
    return quad(integrand, u, np.inf)[0]
#
def get_dpot(u):
    #
    if u != 0:
        out = u * np.exp(u) * expint(u) - 1./3.
    else:
        out = -1./3.
    #
    return out
#
def get_mu(edd,a,l,sig,n0):
	#
	return np.abs(1. - .25*edd*(1+3*np.cos(2*a)) + (128./15.*np.pi)*pow(l,1.5)*np.sqrt(sig)*(1+1.5*edd**2)*pow(n0,0.5))
#
def get_en(psi,U_k,dx,edd,a,l,sig,qf):
	#
	ke = 0
	vdw = 0
	dd = 0
	en = 0
	#
	for x in range(1,len(psi)-1):
		ke = ke + dx*0.5*np.abs((psi[x+1]-psi[x-1])/(2.0*dx))**2
	#
	den = psi*np.conj(psi); n0=den.max();
	mu_0 = get_mu(edd=edd,a=a,l=l,sig=sig,n0=qf*n0)
	#
	qf0 = qf*(128./(15.*np.pi))*pow(sig,0.5)*pow(l,1.5)
	n_k = np.fft.fftshift(fft(den))
	phi_dd = np.fft.ifft(np.fft.ifftshift(U_k*n_k))
	#
	vdw = 0.5*dx*(den*den).sum()
	dd = 0.5*dx*(phi_dd*den).sum()
	qfe = 0.4*qf0*dx*pow(den,2.5).sum()
	en = ke + (vdw + dd + qfe)/(mu_0*n0)
	en = en/(dx*den.sum())
	#
	return np.real(en);
#
def ss_gnd(Dx,Dt,tol,x0,edd,N0,a,l,sig,qf):
	#
	NN = int(1 + 2.0 * Dx[0]/Dx[1])
	xx = np.linspace(-Dx[0],Dx[0],NN)
	Nk = 0.5*(len(xx)-1)
	dk = np.pi/(Dx[1]*Nk)
	kk = np.linspace(-Nk,Nk,NN)*dk
	#
	psi = np.exp(-(xx-x0)**2) + 1e-8*np.random.rand(NN)
	psi = np.sqrt(N0)*psi/np.sqrt(Dx[1]*(psi*np.conj(psi)).sum())
	psi0 = psi
	U = 0
	en = np.random.rand(1)
	en_old = np.random.rand(1)
	en_er = np.random.rand(1)
	en_store = []
	ii = 1
	#
	dpot = np.vectorize(get_dpot)
	U_k = 0.75*edd*(1+3*np.cos(2*a))*dpot(0.5*(kk*sig)**2)
	#
	while en_er > tol:
		#
		den = psi*np.conj(psi); n0 = den.max();
		mu_0 = get_mu(edd=edd,a=a,l=l,sig=sig,n0=qf*n0)
		n_k = np.fft.fftshift(fft(den))
		phi_dd = np.fft.ifft(np.fft.ifftshift(U_k*n_k))
		qft = qf*(128./15.*np.pi)*pow(l,1.5)*pow(sig,0.5)*(1+1.5*edd**2)
		#
		psi = psi*np.exp(-0.5*Dt*(U+(den + phi_dd + qft*pow(den,1.5))/(mu_0*n0)))
		#
		p_k = np.fft.fftshift(fft(psi))/NN
		p_k = p_k*np.exp(-Dt*0.5*kk**2)
		psi = np.fft.ifft(np.fft.ifftshift(p_k))*NN;
		#
		den = psi*np.conj(psi); n0 = den.max();
		mu_0 = get_mu(edd=edd,a=a,l=l,sig=sig,n0=qf*n0)
		n_k = np.fft.fftshift(fft(den))
		phi_dd = np.fft.ifft(np.fft.ifftshift(U_k*n_k))
		qft = qf*(128./15.*np.pi)*pow(l,1.5)*pow(sig,0.5)*(1+1.5*edd**2)
		#
		psi = psi*np.exp(-0.5*Dt*(U+(den + phi_dd + qft*pow(den,1.5))/(mu_0*n0)))
		#
		if divmod(ii,2e4)[1]==0:
			#
			en = get_en(psi=psi,U_k=U_k,dx=Dx[1],edd=edd,a=a,l=l,sig=sig,qf=qf)
			en_er = np.abs((en-en_old)/en)
			en_old = en
			#
			en_store.append(en)
			#
			print('--------------------------')
			print('Norm: ' + repr(Dx[1]*(psi*np.conj(psi)).sum()))
			print('Energy diff: ' + repr(en_er))
			print('Ground state energy ' +repr(en))
			#
		#
		psi = np.sqrt(N0)*psi/np.sqrt(Dx[1]*(psi*np.conj(psi)).sum())
		ii += 1
		#
	#
	print('Itterations: ' + repr(ii-1))
	print('Energy diff: ' + repr(en_er))
	print('Ground state energy: ' + repr(en))
	return psi, en, xx, en_store, phi_dd
#
def ss_rtm(psi,Dx,Dt,T,edd,a,l,sig,qf,ml):
	#
	NN = int(1 + 2.0 * Dx[0]/Dx[1])
	xx = np.linspace(-Dx[0],Dx[0],NN)
	Nk = 0.5*(len(xx)-1)
	dk = np.pi/(Dx[1]*Nk)
	kk = np.linspace(-Nk,Nk,NN)*dk
	#
	U = 0
	NT = T/Dt
	samp = 200
	sptm = np.zeros((samp,NN),dtype=complex)
	ts = np.linspace(0,T,samp)
	jj=1
	c=0
	#
	dpot = np.vectorize(get_dpot)
	U_k = 0.75*edd*(1+3*np.cos(2*a))*dpot(0.5*(kk*sig)**2)
	ml_flag = True
	#
	if T != 0:
		#
		for jj in range(1,int(NT)):
			#
			if jj >= 0.15*NT and ml_flag:
				#
				edd = ml*edd
				U_k = 0.75*edd*(1+3*np.cos(2*a))*dpot(0.5*(kk*sig)**2)
				l = l/ml
				ml_flag = False
				#
			#
			den = psi*np.conj(psi); n0 = den.max();
			mu_0 = get_mu(edd=edd,a=a,l=l,sig=sig,n0=qf*n0)
			n_k = np.fft.fftshift(fft(den))
			phi_dd = np.fft.ifft(np.fft.ifftshift(U_k*n_k))
			qft = qf*(128./15.*np.pi)*pow(l,1.5)*pow(sig,0.5)*(1+1.5*edd**2)
			#
			psi = psi*np.exp(-0.5*1j*Dt*(U+(den + phi_dd + qft*pow(den,1.5))/(mu_0*n0)))
			#
			p_k = np.fft.fftshift(fft(psi))/NN
			p_k = p_k*np.exp(-Dt*1j*0.5*kk**2)
			psi = np.fft.ifft(np.fft.ifftshift(p_k))*NN;
			#
			den = psi*np.conj(psi); n0 = den.max();
			mu_0 = get_mu(edd=edd,a=a,l=l,sig=sig,n0=qf*n0)
			n_k = np.fft.fftshift(fft(den))
			phi_dd = np.fft.ifft(np.fft.ifftshift(U_k*n_k))
			qft = qf*(128./15.*np.pi)*pow(l,1.5)*pow(sig,0.5)*(1+1.5*edd**2)
			#
			psi = psi*np.exp(-0.5*1j*Dt*(U+(den + phi_dd + qft*pow(den,1.5))/(mu_0*n0)))
			#
			if divmod(jj,np.floor(NT/samp))[1]==0:
				#
				sptm[c][:] = psi
				c += 1
				#
			#
			if divmod(jj,np.floor(NT/50))[1]==0:
				#
				print('Real time Progress: ' + repr(np.round(100*jj/NT)) + '%')
				print('edd='+repr(np.float(edd))+' l='+repr(np.float(l)))
				#
			#
		#
	#
	print('--------------------------')
	return psi, sptm, ts
#
Dx = (25.,0.1)
Dt = (1e-4,5e-4)
tol = 1e-10
x0 = 0
edd = 2
N0 = 2e6
a = 0
sig = 0.2
l = 1e-3
T = 400
qf = 1
ml = 1
flag = True
#
t0 = time.time()
this_gnd, this_mu, xx, en_store, phi_dd = ss_gnd(Dx=Dx, Dt=Dt[0], tol=tol, x0=x0, edd=edd, N0=N0, a=a, l=l, sig=sig, qf=qf)
psi_in = this_gnd #+ this_gnd[::-1]*np.exp(0*1j*np.pi)
psi, sptm, ts = ss_rtm(psi=psi_in,Dx=Dx,Dt=Dt[1],T=T,edd=edd,a=a,l=l,sig=sig,qf=qf,ml=ml)
print('Total integration time: '+repr(time.time()-t0))
#
nU = 25.*np.pi*(1-0.25*edd*(1+3*np.cos(2*a)))
nD = 256.*np.sqrt(pow(l,3)*sig)*(1+1.5+edd**2)
print(repr((Dx[1]*nU/nD)**2))
#
if flag == True:
	#
	f1=plt.figure(figsize=(5,3),facecolor='white')
	plt.plot(xx,np.abs(this_gnd)**2)
	plt.xlabel(r"$x$", fontsize=14)
	plt.ylabel(r"$|\psi(x)|^2$", fontsize=14)
	plt.ylim([0,2.0*np.max(np.abs(this_gnd))**2])
	plt.tight_layout()
	f1.show()
	#
	if T != 0:
		#
		f2, ax = plt.subplots(1,1,facecolor='white',figsize=(5,3))
		stp = ax.pcolormesh(ts[0:-1],xx,np.abs(sptm[0:-1,:].T)**2,cmap='Blues')
		#ax.plot(np.linspace(0.15*T,0.15*T,1e2),np.linspace(Dx[0],-Dx[0],1e2),'--',color='white')
		ax.set_ylim([xx[-1],xx[0]])
		ax.set_xlim([0,T])
		plt.suptitle('', fontsize=20)
		plt.xlabel(r"$t$", fontsize=14)
		plt.ylabel(r"$x$", fontsize=14)
		f2.colorbar(stp)
		plt.tight_layout()
		f2.show()
	#
#
