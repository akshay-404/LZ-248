import numpy as np
import json
from rec import *
h = Halo()
E = np.linspace(1, 1200, 4800)
ST = {300: 2.5e-42, 320: 8.5e-42, 340: 5.5e-41, 350: 2.2e-40}
out = {}


def Nacc(m, d, s, skin=1.037): return np.trapezoid(
    spectrum_O1(m, d, s, h, E, skin=skin)*lz_eff(E), E)*2.84


K = {d: (1e-42/Nacc(1000, d, 1e-42))/ST[d]
     for d in ST}          # sigma1_ours/sigma1_ST
print('K(delta) = sigma1(Helm)/sigma1(ST WimPyDD):',
      {d: round(k, 3) for d, k in K.items()})
Kbar = np.mean(list(K.values()))
out['K'] = K
# ---- dipole-charge, natural moment, Helm (same prescription as uploaded script)
MUN = np.sqrt(4*np.pi/137.036)/(2*0.938272)


def Ndip(m, d, mu_muN):
    mu = mu_muN*MUN
    Ek = E*1e-6
    tot = np.zeros_like(E)
    ALPHA = 1/137.036
    for (A, x), w in zip(ISO, MASSFRAC):
        mA = A*AMU
        red = mA*m/(mA+m)
        vv = vmin(Ek, mA, m, d*1e-6)
        coef = Ek*(1/(2*mA)+1/m)+d*1e-6*(1/red+d*1e-6/(2*mA*Ek))
        integ = np.clip(h.at(vv, h.zeta)-coef*h.at(vv, h.eta), 0, None)
        ds = ALPHA*54**2*mu**2/Ek*helm2(A, np.sqrt(2*mA*Ek))*integ
        nT = 1e6/(A*1.66053906660e-24)
        tot += w*nT*h.rho/m*ds*HB2*C*1e5*YR*1e-6
    return np.trapezoid(tot*lz_eff(E), E)*2.84


# neutron-like: mu = 1.91 (m_p/m_B) mu_N
def mu_nat(m): return 1.91*0.938272/m


for m, d in [(1000, 300), (3000, 300), (3000, 330)]:
    nd = Ndip(m, d, mu_nat(m))
    print('dipole m=%d d=%d mu=%.2e muN: N_Helm=%.3f  N_cal(K=%.2f)=%.3f' %
          (m, d, mu_nat(m), nd, Kbar, nd*Kbar))
    out['Ndip_%d_%d' % (m, d)] = (nd, nd*Kbar)
# ---- de-excitation length and survival (LZ event geometry MC)
hbar = 6.582119569e-25


def lam(m, d, v=680.): mu = mu_nat(m)*MUN; G = mu**2 * \
    (d*1e-6)**3/np.pi; return v*1e3*hbar/G


rng = np.random.default_rng(3)
n = 400000
u = rng.normal(size=(n, 3))
u /= np.linalg.norm(u, axis=1)[:, None]
R, H, r0, z0 = 0.728, 1.456, 0.459, 0.264
a = u[:, 0]**2+u[:, 1]**2
b = 2*r0*u[:, 0]
cc = r0**2-R**2
ts = (-b+np.sqrt(b*b-4*a*cc))/(2*a)
tz = np.where(u[:, 2] > 0, (H-z0)/u[:, 2], -z0/u[:, 2])
dexit = np.minimum(ts, tz)
for m, d in [(1000, 300), (2000, 300), (3000, 300), (3000, 330), (4000, 330)]:
    L = lam(m, d)
    P = np.mean(np.exp(-dexit/L))
    P2 = np.mean(np.exp(-(dexit+0.85)/L))
    tau = L/680e3
    print('m=%d d=%d: lambda=%.2f m  tau=%.2e s  P(leave TPC)=%.2f  P(leave all vetoes)=%.2f' % (
        m, d, L, tau, P, P2))
    out['surv_%d_%d' % (m, d)] = (L, tau, P, P2)
# ---- early-universe chi2 depletion: inverse decays track exp(-delta/T) until Gamma*exp(-delta/T) < H
Mpl = 1.22e19
gs = 10.75
for m, d in [(1000, 300), (3000, 330)]:
    tau = lam(m, d)/680e3
    Gam = hbar/tau  # GeV
    Ts = np.logspace(-6, -2, 4000)  # GeV
    Hub = 1.66*np.sqrt(gs)*Ts**2/Mpl
    x = d*1e-6/Ts
    ok = Gam*np.exp(-x) < Hub
    Tdec = Ts[ok].max() if ok.any() else None
    if Tdec is not None:
        print('m=%d d=%d: Gamma=%.2e GeV, T_dec=%.1f keV, f2/f1 at decoupling=%.1e, then decays with tau=%.1e s -> f2(today)=0' %
            (m, d, Gam, Tdec*1e6, np.exp(-d*1e-6/Tdec), tau))
json.dump({k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()})
          for k, v in out.items()}, open('pheno.json', 'w'), indent=1, default=float)
