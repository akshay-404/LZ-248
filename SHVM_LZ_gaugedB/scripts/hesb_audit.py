import numpy as np
from pathlib import Path
import sys

ROOT = lambda x : Path(__file__).resolve().parents[x]
sys.path.insert(0, str(ROOT(2)))

from SHVM_LZ_v3_figures_project.scripts import compare_scenarios as cs

cs.ER = np.linspace(1., 1200., 4797)*1e-6
def flat(er): return 0.96*np.ones_like(er)


def counts(fun, *a, **k):
    acc_save = cs.acceptance
    cs.acceptance = lambda er: np.ones_like(er)      # raw spectrum
    r = fun(*a, **k)
    cs.acceptance = acc_save
    E = cs.ER*1e6
    roi = np.trapezoid(r*acc_save(cs.ER), cs.ER)*cs.EXPOSURE
    he = np.trapezoid(r[(E >= 350) & (E <= 650)]*0.9,
                      cs.ER[(E >= 350) & (E <= 650)])*cs.EXPOSURE
    gap = np.trapezoid(r[(E >= 270) & (E <= 350)],
                       cs.ER[(E >= 270) & (E <= 350)])*cs.EXPOSURE
    peak = E[np.argmax(r)]
    return roi, gap, he, peak


E_ = cs.E
cases = [('A dipole 300', cs.rate_dipole_charge, (1000, 300, 1.8e-3), {}),
         ('A dipole 350', cs.rate_dipole_charge, (1000, 350, 1.8e-3), {}),
         ('B vector 350', cs.rate_vector, (3000, 350, 'B'), dict(eps=1.36e-4)),
         ('C Z 370 xi=0.475', cs.rate_vector,
          (3000, 370, 'Z'), dict(fraction=0.475)),
         ('C Z 275 xi=8.87e-4', cs.rate_vector,
          (3000, 275, 'Z'), dict(fraction=8.87e-4)),
         ('D exo -800 Gp=1e-6', cs.rate_vector,
          (3000, -800, 'D'), dict(mvec=10, eps=1e-6*100/(2*E_))),
         ('D* exo -800 f2=0.1', cs.rate_vector, (3000, -800, 'D'),
          dict(fraction=.1, mvec=1, gdark=.1, eps=5.43e-5)),
         ('E Z+V 290 xi=0.0595', cs.rate_vector, (3000, 290, 'E'), dict(mvec=.15, gdark=.03, node_kev=140, fraction=0.0595))]
print('%-24s %8s %8s %10s %7s' %
      ('case', 'N_ROI', 'N_270-350', 'N_HESB(350-650)', 'peak'))
for lab, f, a, k in cases:
    roi, gap, he, pk = counts(f, *a, **k)
    print('%-24s %8.3f %8.3f %10.3f %7.0f' % (lab, roi, gap, he, pk))
