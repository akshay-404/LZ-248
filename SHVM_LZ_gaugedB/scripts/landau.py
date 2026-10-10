import numpy as np


def pole(weyl, scal, g0, mu0=1440.):
    b = (2/3*weyl+1/3*scal)/(16*np.pi**2)   # beta(g)=b g^3
    return mu0*np.exp(1/(2*b*g0**2))


SM = 3*(6+3+3)*(1/3)**2   # Weyl components x Q^2 : Q_L 6, uc 3, dc 3 per gen


def shvm(NTC, NDTC, nU, nD, q, qX, qXp, qC, BTC):
    w = SM
    w += NTC*(2*BTC**2)+NTC*2*BTC**2          # Q_TC (2 comps) and T_R, B_R
    w += 2*3*(nU+nD)*q**2 + 2*3*NTC*qX**2 + 2*3*NDTC*qXp**2 + 2*NDTC*2*qC**2
    return w


S2 = 15**2
for lab, args, g in [('SHVM (2, 2), nU, nD=(3, 6)', (2, 2, 3, 6, 2.5, 1, -2, -4.5, -1.5), 0.191),
                     ('SHVM (4, 2), nU, nD=(3, 6)',
                      (4, 2, 3, 6, 2.5, 1.75, -4.5, -7, -0.75), 0.191),
                     ('SHVM (2, 2), nU, nD=(3, 3)', (2, 2, 3, 3, 2.5, 1, -9.5, -12, -1.5), 0.191)]:
    w = shvm(*args)
    print('%-26s sum_Weyl Q^2=%6.1f  Landau pole = %.0f TeV' %
          (lab, w, pole(w, S2, g)/1e3))
# Sannino-Turner charges: U, D (3 colours, B=5/2), spectators Psi_L, R doublets (B1=-1, B2=2), eta, N; scalars S(-15), Phi(-3)
wST = SM+2*3*2*(2.5)**2 + 2*(1+4) + (1+4) + (1+4)
print('%-26s sum_Weyl Q^2=%6.1f  Landau pole = %.0f TeV (gB=0.146) / %.0f TeV (gB=0.20)' %
      ('Sannino-Turner', wST, pole(wST, S2+9, 0.146)/1e3, pole(wST, S2+9, 0.20)/1e3))
