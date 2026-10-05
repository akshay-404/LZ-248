import numpy as np
from rec import *
h = Halo()
E = np.linspace(1, 1200, 4800)
K = 0.394
HB2 = 0.389379e-27
mp = 0.938272


def parts(m, d, s, halo=h, skin=1.037):
    r = spectrum_O1(m, d, s, halo, E, skin=skin)
    roi = np.trapezoid(r*lz_eff(E), E)*2.84*K
    w = (E >= 350) & (E <= 650)
    he = np.trapezoid(r[w], E[w])*0.9*2.84*K
    return r, roi, he


def portal(sigma, m, Qchi=7.5):
    mup = m*mp/(m+mp)
    GN = np.sqrt(np.pi*sigma/HB2)/mup
    return np.sqrt(Qchi/GN)   # M_V/g_B in GeV


for lab, m, d, s in [
    ('BM1 (ST portal)', 1000, 300, 6.5e-43),
    ('BM2 (SHVM)', 3000, 300, 1.9e-42),
    ('BM2* (one-event)', 3000, 300, 3.78e-42)
]:
    r, N, he = parts(m, d, s)
    MG = portal(s, m)
    g = 1440/MG
    vS = 1440/(15*g)
    Nj = parts(m, d, s, Halo(vE=265.))[1]
    Nd = parts(m, d, s, Halo(vE=235.))[1]
    sup = E[r > 0]
    print('%-16s N_acc=%.3f N_HE=%.3f M_V/g_B=%.3f TeV g_B=%.3f g_q=%.3f v_S=%.3f GeV pert=%.3f Jun/Dec=%.3f support %.1f-%.1f keV per t*yr=%.3f' % 
          (lab, N, he, MG/1e3, g, g/3, vS, (15*g)**2/(4*np.pi), Nj/Nd, sup.min(), sup.max(), N/2.84))

# Lambda_* (ST eq.18 scaling)
for vS, LD in [(0.502, 1.0), (0.502, 3.0), (0.657, 1.0)]:
    print('Lambda* = %.3f TeV (v_S=%.3f TeV, Lambda_D=%.0f TeV)' %
          (13*vS**(1/6)*LD, vS, LD))

# EM mass ordering of DQCD baryons: E_EM ~ alpha <1/r> sum_{i<j} Q_i Q_j
a = 1/137.036
for inv in (0.5e3, 1e3):
    print('<1/r>=%.1f TeV: UUD-UDD=%.2f GeV, DDD-UDD=%.2f GeV (plus mass differences)' %
          (inv/1e3, a*inv/3, 2*a*inv/3))

# technibaryon (B=-3) elastic V cross-section vs BM2 and Boltzmann estimate
s_TB = (3/7.5)**2*1.9e-42
for mTB in (0.3e3, 1e3, 2.5e3, 4e3):
    x = mTB/150.
    OmegaRatio = (mTB/mp)*x**1.5*np.exp(-x)
    xi = OmegaRatio/5.36
    print('m_TB=%.1f TeV: sigma_TB=%.3e cm^2, xi_TB~%.3e, xi*sigma=%.1e vs limit~%.0e' %
          (mTB/1e3, s_TB, xi, xi*s_TB, 1e-46*mTB/1e3))
