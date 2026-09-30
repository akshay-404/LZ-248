import numpy as np
import importlib.util
import json
from SHVM_LZ_v3_figures_project.scripts import compare_scenarios as cs

cs.ER = np.linspace(1., 1300., 5197)*1e-6
K = 0.394
acc0 = cs.acceptance


def raw(fun, *a, **k):
    cs.acceptance = lambda er: np.ones_like(er)
    r = fun(*a, **k)
    cs.acceptance = acc0
    return r


def stats(r):
    E = cs.ER*1e6
    ra = r*acc0(cs.ER)
    roi = np.trapezoid(ra, cs.ER)*cs.EXPOSURE

    def f(lo, hi, w): return np.trapezoid(
        w[(E >= lo) & (E <= hi)], cs.ER[(E >= lo) & (E <= hi)])*cs.EXPOSURE
    return dict(roi=roi, f200=f(200, 300, ra)/roi, fl150=f(0, 150, ra)/roi, gap=f(270, 350, r)/roi, he=0.9*f(350, 650, r)/roi, peak_raw=E[np.argmax(r)])


Ee = cs.E
out = {}
for lab, f, a, k in [('D-270', cs.rate_vector, (3000, -270, 'D'), dict(mvec=10, eps=1e-6*100/(2*Ee))),
                     ('D-600', cs.rate_vector, (3000, -600, 'D'),
                      dict(mvec=10, eps=1e-6*100/(2*Ee))),
                     ('C350', cs.rate_vector, (3000, 350, 'Z'), {}),
                     ('A300', cs.rate_dipole_charge, (1000, 300, 1.8e-3), {}),
                     ('B350', cs.rate_vector, (3000, 350, 'B'), dict(eps=1e-4)),
                     ('C275', cs.rate_vector, (3000, 275, 'Z'), {}),
                     ('C370', cs.rate_vector, (3000, 370, 'Z'), {}),
                     ('D-800', cs.rate_vector, (3000, -800, 'D'),
                      dict(mvec=10, eps=1e-6*100/(2*Ee))),
                     ('E290', cs.rate_vector, (3000, 290, 'E'), dict(mvec=.15, gdark=.03, node_kev=140))]:
    s = stats(raw(f, *a, **k))
    out[lab] = s
    print('%-6s N1(Helm,param ref)=%10.4g  f200-300=%.2f  f<150=%.2f  R_gap=%.3f  R_HE=%.3f  rawpeak=%.0f' %
          (lab, s['roi'], s['f200'], s['fl150'], s['gap'], s['he'], s['peak_raw']))
# C hybrid at relic fraction xi=(1.4-9)e-4 (v4): find delta with calibrated N=1
for xi in (1.4e-4, 9e-4):
    ds = np.arange(200, 300, 1.)
    # rate_vector already includes acceptance
    N = [np.trapezoid(cs.rate_vector(3000, float(d), 'Z'), cs.ER)
         * cs.EXPOSURE*K*xi for d in ds]
    d1 = np.interp(0, -np.log(N), ds)
    s = stats(raw(cs.rate_vector, 3000, d1, 'Z'))
    print('C at xi=%.1e: delta(N_cal=1)=%.0f keV, f200-300=%.2f f<150=%.2f R_HE=%.3f' %
          (xi, d1, s['f200'], s['fl150'], s['he']))
    out['Crelic_%g' % xi] = dict(delta=d1, **s)
json.dump(out, open('cmp.json', 'w'), indent=1, default=float)
