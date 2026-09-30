import matplotlib.pyplot as plt
from rec import *
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
h = Halo()
E = np.linspace(1, 1200, 4800)
K = 0.394
HB2 = 0.389379e-27
mp = 0.938272
rows = json.load(open('bench.json'))


def sig_of_MG(MG, m, Q=7.5):
    mup = m*mp/(m+mp)
    G = Q/MG**2
    return mup**2*G**2/np.pi*HB2


MV = 1440.
sig_pert = sig_of_MG(15*MV/np.sqrt(4*np.pi), 1000.)
sig_dij = sig_of_MG(MV/(3*0.084), 1000.)
fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
for m, col, lab in ((1000, 'C0', '1 TeV'), (3000, 'C3', '3 TeV')):
    R = [r for r in rows if r['m'] == m]
    d = np.array([r['d'] for r in R])
    s1 = np.array([r['sigma1'] for r in R])
    sh = np.array([r['sigHEmax'] for r in R])
    if m == 1000:
        ax[0].fill_between(d, 0.144*s1, 2.88*s1, color=col, alpha=0.18,
                           label=r'LZ 90% interval (1 TeV; ST digitisation, scaled)')
    ax[0].plot(d, s1, '-', color=col, label=r'$N_{\rm acc}=1$, %s' % lab)
    ax[0].plot(d, sh, '--', color=col,
               label=r'HE-sideband bound $N_{\rm HE}<2.3$, %s' % lab)
ax[0].axhline(sig_pert, color='k', ls=':',
              label=r'$(15g_B)^2/4\pi=1$ at $M_V=1.44$ TeV')
ax[0].axhline(sig_dij, color='gray', ls='-.',
              label=r'dijet $g_q<0.084$ at $M_V=1.44$ TeV')
ax[0].plot([300], [9.9e-43], 'v', color='k', ms=7,
           label=r'solar $\nu$ ($b\bar b$), 1 TeV')
ax[0].plot([300], [6.5e-43], '*', color='gold',
           mec='k', ms=13, label='ST benchmark')
ax[0].plot([300], [1.9e-42], 'P', color='C3',
           mec='k', ms=10, label='SHVM BM2 (3 TeV)')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'$\delta$ [keV]')
ax[0].set_ylabel(r'$\sigma_p^0$ [cm$^2$]')
ax[0].set_ylim(1e-43, 3e-39)
ax[0].legend(fontsize=6.5, loc='upper center',
             bbox_to_anchor=(0.5, -0.16), ncol=2)
ax[0].set_title('(a) isoscalar vector portal: constraint plane')
for d, c in ((300, 'C0'), (320, 'C1'), (340, 'C2')):
    r = spectrum_O1(3000, d, 1e-42, h, E, skin=1.037)
    n = np.trapezoid(r*lz_eff(E), E)*2.84
    r = r/n*2.84
    ax[1].plot(E, r*lz_eff(E), '-', color=c,
               label=r'$\delta=%d$ keV (accepted)' % d)
    ax[1].plot(E, r, ':', color=c)
ax[1].axvspan(350, 650, color='k', alpha=0.08,
              label='LZ HE sideband (0 events)')
ax[1].axvspan(225, 271, color='orange', alpha=0.25,
              label=r'event $\pm1\sigma_{\rm stat}$')
ax[1].set_xlim(80, 700)
ax[1].set_xlabel(r'true recoil energy [keV]')
ax[1].set_ylabel(r'$dN/dE_R$ [keV$^{-1}$], one accepted event')
ax[1].legend(fontsize=7)
ax[1].set_title(r'(b) $m_\chi=3$ TeV spectra (dotted: before acceptance)')
P = json.load(open('pheno.json'))
ms = np.array([0.7, 1, 1.5, 2, 3, 4, 5])*1000
MUN = np.sqrt(4*np.pi/137.036)/(2*mp)
hbar = 6.582119569e-25
rng = np.random.default_rng(3)
n = 200000
u = rng.normal(size=(n, 3))
u /= np.linalg.norm(u, axis=1)[:, None]
Rr, H, r0, z0 = 0.728, 1.456, 0.459, 0.264
a = u[:, 0]**2+u[:, 1]**2
b = 2*r0*u[:, 0]
cc = r0**2-Rr**2
dexit = np.minimum((-b+np.sqrt(b*b-4*a*cc))/(2*a),
                   np.where(u[:, 2] > 0, (H-z0)/u[:, 2], -z0/u[:, 2]))
for d, ls in ((300, '-'), (330, '--')):
    L = [680e3*hbar*np.pi/(((1.91*mp/m)*MUN)**2*(d*1e-6)**3) for m in ms]
    ax[2].plot(ms/1000, [np.mean(np.exp(-dexit/l)) for l in L], ls,
               color='C0', label=r'leave TPC, $\delta=%d$ keV' % d)
    ax[2].plot(ms/1000, [np.mean(np.exp(-(dexit+0.85)/l)) for l in L],
               ls, color='C3', label=r'clear all vetoes, $\delta=%d$ keV' % d)
ax[2].set_xlabel(r'$m_\chi$ [TeV]')
ax[2].set_ylabel('single-scatter survival probability')
ax[2].set_ylim(0, 1)
ax[2].legend(fontsize=7)
ax[2].set_title(r'(c) SHVM $\chi_2\to\chi_1\gamma$ self-veto at the LZ event')
plt.tight_layout()
plt.savefig('fig_shvm_st.pdf')
plt.savefig('fig_shvm_st.png', dpi=80)
print('sig_pert=%.2e sig_dijet=%.2e' % (sig_pert, sig_dij))
