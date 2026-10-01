import numpy as np
import json
from rec import *
from pathlib import Path

ROOT = lambda x : Path(__file__).resolve().parents[x]

h = Halo()
E = np.linspace(1, 1200, 4800)
K = 0.394
mp = 0.938272
HB2 = 0.389379e-27


def parts(m, d, skin=1.037, halo=h):
    r = spectrum_O1(m, d, 1e-42, halo, E, skin=skin)
    roi = np.trapezoid(r*lz_eff(E), E)*2.84
    w = (E >= 350) & (E <= 650)
    he = np.trapezoid(r[w], E[w])*0.9*2.84
    return roi, he, r


def portal(sigma, m, Qchi=7.5):
    mup = m*mp/(m+mp)
    GN = np.sqrt(np.pi*sigma/HB2)/mup
    return np.sqrt(Qchi/GN)   # M_V/g_B in GeV


rows = []
print('%5s %4s %10s %10s %7s %8s %8s %7s %7s %7s' % ('m', 'd', 'sig1_cal',
      'sigHEmax', 'R_HE', 'MV/gB', 'g_q@1.44', 'pert', 'Jun/Dec', 'f200'))
hJ = Halo(vE=265.)
hD = Halo(vE=235.)
for m in (1000., 3000.):
    for d in (280, 300, 310, 320, 330, 340, 350, 360):
        roi, he, r = parts(m, d)
        if roi <= 0:
            continue
        s1 = 1e-42/roi/K          # calibrated one-accepted-event cross section
        R = he/roi
        sHE = 2.3*s1/R if R > 0 else np.inf   # N_HE<2.3 (0 observed)
        MG = portal(s1, m)
        gq = 1440/MG/3
        pert = (15*1440/MG)**2/(4*np.pi)
        mod = parts(m, d, halo=hJ)[0]/parts(m, d, halo=hD)[0]
        acc = r*lz_eff(E)
        f200 = np.trapezoid(acc[E >= 200], E[E >= 200])/np.trapezoid(acc, E)
        rows.append(dict(m=m, d=d, sigma1=s1, sigHEmax=sHE, R_HE=R,
                    MVgB=MG, gq=gq, pert=pert, mod=mod, f200=f200))
        print('%5.0f %4d %10.2e %10.2e %7.3f %8.0f %8.3f %7.2f %7.2f %7.2f' %
              (m, d, s1, sHE, R, MG, gq, pert, mod, f200))
json.dump(rows, open(ROOT(0) / 'bench.json', 'w'), indent=1)
# ST benchmark reproduction with calibration: sigma=6.5e-43 at 1 TeV 300 keV
roi, he, r = parts(1000, 300)
print('ST benchmark sigma=6.5e-43: N_acc(cal)=%.3f (ST: 0.26), N_HE=%.4f' %
      (roi*6.5e-43/1e-42*K, he*6.5e-43/1e-42*K))
print('v_S for MV=1.44 TeV, gB=0.146, Q_S=15:', 1440/(15*0.146), 'GeV')
