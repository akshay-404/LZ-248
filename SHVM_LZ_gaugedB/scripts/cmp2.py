from rec import *
import numpy as np
import json
from pathlib import Path
import sys

ROOT = lambda x : Path(__file__).resolve().parents[x]
sys.path.insert(0, str(ROOT(2)))

from SHVM_LZ_v3_figures_project.scripts import compare_scenarios as cs

cs.ER = np.linspace(1., 1300., 5197)*1e-6
K = 0.394
acc0 = cs.acceptance
out = json.load(open(ROOT(0) / 'cmp.json'))

for xi in (1.4e-4, 9e-4):
    ds = np.arange(120, 300, 1.)
    N = np.array([np.trapezoid(cs.rate_vector(3000, float(d), 'Z'), cs.ER)*cs.EXPOSURE*K*xi for d in ds])
    d1 = float(np.interp(0, -np.log(N), ds))
    cs.acceptance = lambda er: np.ones_like(er)
    r = cs.rate_vector(3000, d1, 'Z')
    cs.acceptance = acc0
    E = cs.ER*1e6
    ra = r*acc0(cs.ER)
    roi = np.trapezoid(ra, cs.ER)

    def g(lo, hi, w):
        return np.trapezoid(w[(E >= lo) & (E <= hi)], cs.ER[(E >= lo) & (E <= hi)])/roi
    
    print('C xi=%.1e: delta=%.0f keV f200-300=%.2f f<150=%.2f R_HE=%.3f' %
          (xi, d1, g(200, 300, ra), g(0, 150, ra), 0.9*g(350, 650, r)))
    out['Crelic_%g' % xi] = dict(delta=d1, f200=g(200, 300, ra), fl150=g(0, 150, ra), he=0.9*g(350, 650, r))

h = Halo()
Ee = np.linspace(1, 1300, 5200)
for m in (1000., 3000.):
    r = spectrum_O1(m, 300, 1e-42, h, Ee, skin=1.037)
    ra = r*lz_eff(Ee)
    roi = np.trapezoid(ra, Ee)

    def g(lo, hi, w): return np.trapezoid(
        w[(Ee >= lo) & (Ee <= hi)], Ee[(Ee >= lo) & (Ee <= hi)])/roi
    print('G m=%.0f d=300: f200-300=%.2f f<150=%.2f R_gap=%.3f R_HE=%.3f' %
          (m, g(200, 300, ra), g(0, 150, ra), g(270, 350, r), 0.9*g(350, 650, r)))
    out['G_%d' % m] = dict(f200=g(200, 300, ra), fl150=g(
        0, 150, ra), gap=g(270, 350, r), he=0.9*g(350, 650, r))
json.dump(out, open(ROOT(0) / 'cmp2.json', 'w'), indent=2, default=float)
