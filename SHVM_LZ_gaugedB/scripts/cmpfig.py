import matplotlib.pyplot as plt
from rec import *
import numpy as np
import matplotlib
matplotlib.use('Agg')

from pathlib import Path
import sys

ROOT = lambda x : Path(__file__).resolve().parents[x]
sys.path.insert(0, str(ROOT(2)))

from SHVM_LZ_v3_figures_project.scripts import compare_scenarios as cs

cs.ER = np.linspace(1., 1100., 4397)*1e-6
acc0 = cs.acceptance
Ee = cs.E

def curve(fun, *a, **k):
    cs.acceptance = lambda er: np.ones_like(er)
    r = fun(*a, **k)
    cs.acceptance = acc0
    n = np.trapezoid(r*acc0(cs.ER), cs.ER)*cs.EXPOSURE
    return cs.ER*1e6, r*cs.EXPOSURE/n*1e-6


cases = [('A dipole, 1 TeV, 300 keV', cs.rate_dipole_charge, (1000, 300, 1.8e-3), {}, 'C0', '-'),
         ('B dark vector, 3 TeV, 350 keV', cs.rate_vector,
          (3000, 350, 'B'), dict(eps=1e-4), 'C1', '-'),
         ('C hybrid Z, 3 TeV, 275 keV', cs.rate_vector,
          (3000, 275, 'Z'), {}, 'C2', '-'),
         ('C hybrid Z, 3 TeV, 370 keV', cs.rate_vector,
          (3000, 370, 'Z'), {}, 'C2', '--'),
         ('D exothermic, 3 TeV, $-800$ keV', cs.rate_vector,
          (3000, -800, 'D'), dict(mvec=10, eps=1e-6*100/(2*Ee)), 'C3', '-'),
         ('D exothermic, 3 TeV, $-270$ keV', cs.rate_vector,
          (3000, -270, 'D'), dict(mvec=10, eps=1e-6*100/(2*Ee)), 'C3', ':'),
         ('E hybrid $Z{+}V$, 3 TeV, 290 keV', cs.rate_vector, (3000, 290, 'E'), dict(mvec=.15, gdark=.03, node_kev=140), 'C4', '-')]
fig, ax = plt.subplots(figsize=(11, 4.6))
for lab, f, a, k, c, ls in cases:
    x, y = curve(f, *a, **k)
    ax.plot(x, y, ls, color=c, label=lab, lw=1.3)
h = Halo()
E = np.linspace(1, 1100, 4400)
r = spectrum_O1(3000, 300, 1e-42, h, E, skin=1.037)
n = np.trapezoid(r*lz_eff(E), E)
ax.plot(E, r/n, 'k-', lw=2.2, label=r'gauged-$B$ model (BM2), 3 TeV, 300 keV')
ax.axvspan(5.4, 270, color='C0', alpha=0.05)
ax.axvspan(350, 650, color='k', alpha=0.10)
ax.axvspan(225, 271, color='orange', alpha=0.25)
ax.axvspan(271, 350, color='gold', alpha=0.08)
ax.text(100, 3e-2, 'LZ ROI', fontsize=8)
ax.text(430, 3e-2, 'HE sideband\n(0 events)', fontsize=8)
ax.text(228, 5e-2, 'event', fontsize=8)
ax.text(285, 3e-2, 'gap', fontsize=8)
ax.set_yscale('log')
ax.set_ylim(1e-6, 8e-2)
ax.set_xlim(0, 1000)
ax.set_xlabel('true recoil energy [keV]')
ax.set_ylabel(r'$dN/dE_R$ before acceptance [keV$^{-1}$]')
ax.set_title('All scenarios normalised to one accepted ROI event', fontsize=10)
ax.legend(fontsize=7.5, loc='upper left', bbox_to_anchor=(1.01, 1.0))
plt.tight_layout()
plt.savefig(ROOT(1) / 'fig_scenarios.pdf')
plt.savefig(ROOT(1)  / 'fig_scenarios.png', dpi=90)
