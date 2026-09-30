"""Isotope-resolved endothermic/exothermic recoil pipeline (SHM, Helm)."""
import numpy as np
from scipy.special import erf, erfc, spherical_jn
from scipy.integrate import cumulative_trapezoid
C=299792.458; HB2=0.389379e-27; YR=31557600.; AMU=0.9314941
ISO=[(124,.00095),(126,.00089),(128,.0191),(129,.26401),(130,.04071),(131,.21232),(132,.26909),(134,.10436),(136,.08857)]
MASSFRAC=np.array([a*f for a,f in ISO]); MASSFRAC/=MASSFRAC.sum()
class Halo:
    def __init__(s,v0=238.,vesc=544.,vE=250.2,rho=0.3):
        s.v0,s.vesc,s.vE,s.rho=v0,vesc,vE,rho
        s.vs=np.linspace(1e-4,vesc+vE,20001)
        z=vesc/v0; norm=erf(z)-2*z*np.exp(-z*z)/np.sqrt(np.pi)
        v=s.vs; pref=v/(np.sqrt(np.pi)*v0*vE*norm)
        pdf=pref*(np.exp(-((v-vE)/v0)**2)-np.where(v<vesc-vE,np.exp(-((v+vE)/v0)**2),np.exp(-z*z)))
        s.pdf=np.clip(pdf,0,None)
        s.eta=-cumulative_trapezoid((s.pdf/(v/C))[::-1],v[::-1],initial=0)[::-1]   # int f/v (v in c)
        s.zeta=-cumulative_trapezoid((s.pdf*(v/C))[::-1],v[::-1],initial=0)[::-1]
    def at(s,vm,arr): return np.interp(vm,s.vs,arr,right=0.)
def helm2(A,q,scale=1.0):
    qf=q/0.1973269804; c=1.23*A**(1/3)-0.60; a,sk=.52,.9
    rn=np.sqrt(c*c+7*np.pi**2*a*a/3-5*sk*sk)*scale; x=np.maximum(qf*rn,1e-12)
    return (3*spherical_jn(1,x)/x)**2*np.exp(-(qf*sk)**2)
def lz_eff(E_keV):
    low=1/(1+np.exp(-np.clip((E_keV-5.4)/1.5,-100,100))); high=0.5*erfc((E_keV-269.9)/(12*np.sqrt(2)))
    return 0.96*low*high
def vmin(E,mA,m,d):
    mu=mA*m/(mA+m); return np.abs(mA*E/mu+d)/np.sqrt(2*mA*E)*C
def spectrum_O1(m,delta_keV,sigma_p,halo,E_keV,fn_fp=1.0,skin=1.0,Z=54):
    """dR/dE [events / tonne / yr / keV]; isoscalar-like O1 with nucleon couplings fp=1,fn=fn_fp.
    sigma_p = zero-momentum per-proton cross section (cm^2)."""
    E=E_keV*1e-6; out=np.zeros_like(E); mp=0.938272; mup=m*mp/(m+mp)
    for (A,x) in ISO:
        mA=A*AMU; N=A-Z
        q=np.sqrt(2*mA*E)
        Fp2=helm2(A,q); Fn2=helm2(A,q,skin)
        amp2=(Z*np.sqrt(Fp2)*np.sign(1)+fn_fp*N*np.sqrt(Fn2))**2
        eta=halo.at(vmin(E,mA,m,delta_keV*1e-6),halo.eta)
        # dsigma/dE = mA sigma_p amp2/(2 mup^2 v^2)
        rate=halo.rho/m*mA*(sigma_p/HB2)*amp2/(2*mup**2)*eta   # per nucleus per unit E (GeV) in natural*c units
        nT=1e6/(A*1.66053906660e-24)
        out+=np.array(MASSFRAC[[a for a,_ in ISO].index(A)])*nT*rate*HB2*C*1e5*YR*1e-6  # per keV
    return out
