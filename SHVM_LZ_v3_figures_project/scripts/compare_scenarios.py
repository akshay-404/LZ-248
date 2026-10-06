"""Auditable recoil-only SHVM scenario comparison.

No LZ S1/S2 response, veto propagation, nuclear-spin response, profile
likelihood, relic computation, or experimental exclusion recast is included.
The natural xenon isotope mixture, halo and efficiency are explicit.
"""
from __future__ import annotations

import json
from pathlib import Path
ROOT = lambda x : Path(__file__).resolve().parents[x]

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf, erfc, spherical_jn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = 299792.458  # km/s
GF = 1.1663787e-5  # GeV^-2
ALPHA = 1 / 137.036
E = np.sqrt(4 * np.pi * ALPHA)
SW2 = 0.2312
HBARC2 = 0.389379e-27  # cm^2 GeV^2
YR = 31557600.0  # s
RHO = 0.3  # GeV/cm^3
EXPOSURE = 2.84  # tonne year
PROTON = 0.938272  # GeV
MUN = E / (2 * PROTON)  # nuclear magneton in GeV^-1
ISOTOPES = [(124, .00095), (126, .00089), (128, .01910),
            (129, .26401), (130, .04071), (131, .21232),
            (132, .26909), (134, .10436), (136, .08857)]
ABAR = sum(a * f for a, f in ISOTOPES)
N_TONNE = 1e6 / (ABAR * 1.66053906660e-24)
V0, VESC, VE = 238., 544., 250.  # km/s
VS = np.linspace(1e-5, VESC + VE, 15001)
ER = np.linspace(1., 399., 1593) * 1e-6  # true recoil, GeV


def speed_pdf(v: np.ndarray) -> np.ndarray:
    z = VESC / V0
    norm = erf(z) - 2 * z * np.exp(-z*z) / np.sqrt(np.pi)
    pref = v / (np.sqrt(np.pi) * V0 * VE * norm)
    first = np.exp(-((v - VE)/V0)**2)
    second = np.where(v < VESC - VE, np.exp(-((v + VE)/V0)**2), np.exp(-z*z))
    return np.where((v >= 0) & (v <= VESC + VE), pref * (first-second), 0.)


PDF = speed_pdf(VS)
assert abs(np.trapezoid(PDF, VS) - 1) < 2e-4


def reverse_integral(values: np.ndarray) -> np.ndarray:
    return -cumulative_trapezoid(values[::-1], VS[::-1], initial=0)[::-1]


ETA = reverse_integral(PDF / (VS/C))
ZETA = reverse_integral(PDF * (VS/C))


def helm(a: int, q: np.ndarray) -> np.ndarray:
    qfm = q / 0.1973269804
    c = 1.23 * a**(1/3) - .60
    skin, diffuseness = .90, .52
    rn = np.sqrt(c*c + 7*np.pi**2*diffuseness**2/3 - 5*skin**2)
    x = np.maximum(qfm*rn, 1e-12)
    return 3*spherical_jn(1, x)/x*np.exp(-(qfm*skin)**2/2)


def acceptance(er: np.ndarray) -> np.ndarray:
    ek = er*1e6
    low = 1/(1 + np.exp(-np.clip((ek-5.4)/1.5, -100, 100)))
    high = erfc((ek-269.9)/(12*np.sqrt(2)))/2
    return .96*low*high


def vmin(er: np.ndarray, ma: float, mass: float, delta: float) -> np.ndarray:
    mu = ma*mass/(ma+mass)
    return np.abs(ma*er/mu+delta)/np.sqrt(2*ma*er)*C


def moment(v: np.ndarray, integrand: np.ndarray) -> np.ndarray:
    return np.interp(v, VS, integrand, right=0)


def common_rate_factor(mass: float, fraction: float) -> float:
    return fraction*N_TONNE*YR*RHO/mass*C*1e5*HBARC2


def rate_vector(
        mass: float,
        delta_kev: float,
        kind: str,
        fraction: float = 1,
        mvec: float = 10,
        gdark: float = 1,
        eps: float = 1e-4,
        node_kev: float = 110) -> np.ndarray:
    """Vector amplitude with kind = Z, B, D, E; E = hybrid Z+V."""
    delta = delta_kev*1e-6  # negative for exothermic D
    out = np.zeros_like(ER)
    fn = np.sqrt(2)*GF*.25
    fp = -(1-4*SW2)*fn
    for a, abund in ISOTOPES:
        ma = .9315*a
        q2 = 2*ma*ER
        ff = helm(a, np.sqrt(q2))
        z, n = 54, a-54
        if kind == 'Z':
            amp = (n*fn+z*fp)*ff
        elif kind == 'B':
            amp = 3*gdark*eps*E*z*ff/(q2+mvec*mvec)
        elif kind == 'D':
            amp = 2*gdark*eps*E*z*ff/(q2+mvec*mvec)
        elif kind == 'E':
            # The node is tuned only for A=131.3 and equal p/n Helm response.
            A0 = 131.3
            ma0 = A0*0.9315
            n0 = A0-z
            f0 = n0*fn+z*fp
            eps_node = -f0*(2*ma0*node_kev*1e-6 + mvec*mvec)/(3*gdark*E*54)
            amp = (n*fn+z*fp+3*gdark*eps_node*E*z/(q2+mvec*mvec))*ff
        else:
            raise ValueError(kind)
        eta = moment(vmin(ER, ma, mass, delta), ETA)
        out += abund * ma*amp**2/(2*np.pi)*eta
    return common_rate_factor(mass, fraction)*out*acceptance(ER)


def rate_dipole_charge(mass: float, delta_kev: float, dipole_muN: float, fraction: float = 1) -> np.ndarray:
    """Leading electric-charge nuclear response of a transition dipole.

    The positive magnetic-magnetic nuclear-spin contribution is omitted.
    Equation (6) of arXiv:1007.4200 supplies the bracket and spin-1/2 factor.
    """
    delta = delta_kev*1e-6
    mu_dm = dipole_muN*MUN
    out = np.zeros_like(ER)
    for a, abund in ISOTOPES:
        ma = a*.9315
        reduced = ma*mass/(ma+mass)
        vv = vmin(ER, ma, mass, delta)
        eta = moment(vv, ETA)
        zeta = moment(vv, ZETA)
        coefficient = ER*(1/(2*ma)+1/mass)+delta*(1/reduced+delta/(2*ma*ER))
        integrated = np.maximum(zeta-coefficient*eta, 0)
        ff = helm(a, np.sqrt(2*ma*ER))
        out += abund*ALPHA*54**2*mu_dm**2/ER*ff**2*integrated
    return common_rate_factor(mass, fraction)*out*acceptance(ER)


def integral(rate: np.ndarray, lo: float = 5.4, hi: float = 270) -> float:
    mask = (ER*1e6 >= lo) & (ER*1e6 <= hi)
    return float(np.trapezoid(rate[mask], ER[mask]))


def row(label: str, rate: np.ndarray, mass: float, delta: float) -> dict:
    full = integral(rate)*EXPOSURE
    total = integral(rate)
    return {
        'scenario': label,
        'mass_GeV': mass,
        'delta_keV_signed': delta,
        'N_toy_per_2.84_tonneyear': full,
        'fraction_200_270_keV': integral(rate, 200, 270)/total if total else None,
        'fraction_below_150_keV': integral(rate, 5.4, 150)/total if total else None,
        'peak_keV': float(ER[np.argmax(rate)]*1e6),
        'vmin_248_kms_A131': float(vmin(np.array([248e-6]), 131*.9315, mass, delta*1e-6)[0]),
    }


def decay_length(mass: float, delta_kev: float, dipole_muN: float, speed_kms: float = 680) -> float:
    mu_dm = dipole_muN*MUN
    width = mu_dm**2*(delta_kev*1e-6)**3/np.pi
    return speed_kms * 1000 * 6.582119569e-25/width


def sensitivity_scan() -> dict:
    """Change one surrogate input at a time, restoring globals afterward."""
    global VESC, VS, PDF, ETA, ZETA, helm
    saved = (VESC, VS, PDF, ETA, ZETA, helm)

    def selected_counts() -> dict:
        return {
            'B_delta350_eps1e-4':integral(rate_vector(3000, 350, 'B'))*EXPOSURE,
            'C_delta370_qZquarter':integral(rate_vector(3000, 370, 'Z'))*EXPOSURE,
            'D_minus800_Gp1e-6':integral(rate_vector(3000, -800, 'D', eps=1e-6*100/(2*E)))*EXPOSURE,
            'E_delta290_node140':integral(rate_vector(3000, 290, 'E', mvec=.15, gdark=.03, node_kev=140))*EXPOSURE,
        }

    output = {'vesc_kms':{}, 'uniform_Helm_q_scale':{}}
    try:
        for esc in (511., 544., 577.):
            VESC = esc
            VS = np.linspace(1e-5, VESC+VE, 15001)
            PDF = speed_pdf(VS)
            ETA = reverse_integral(PDF/(VS/C))
            ZETA = reverse_integral(PDF*(VS/C))
            output['vesc_kms'][str(int(esc))] = selected_counts()
        VESC, VS, PDF, ETA, ZETA = saved[:5]
        original_helm = saved[5]
        for scale in (.97, 1., 1.03):
            helm = lambda a, q, s=scale: original_helm(a, q*s)
            output['uniform_Helm_q_scale'][str(scale)] = selected_counts()
    finally:
        VESC, VS, PDF, ETA, ZETA, helm = saved
    return output


def main() -> None:

    rows = []
    for d in [300, 350]:
        rA = rate_dipole_charge(1000, d, 1.8e-3)
        line = row('A dipole-charge, mu=0.0018muN, xi=1', rA, 1000, d)
        lam = decay_length(1000, d, 1.8e-3)
        line['lambda_m_at_680_kms'] = lam
        line['toy_survival_L0p5m'] = float(np.exp(-.5/lam))
        line['toy_max_science_events_L0p5m'] = line['N_toy_per_2.84_tonneyear']*lam/(np.e*.5)
        rows.append(line)

    rB = rate_vector(3000, 350, 'B')
    bn = integral(rB)*EXPOSURE
    eps_one = 1e-4/np.sqrt(bn)
    rows.append(row('B UDD vector, eps=1e-4, g=1, mV=10GeV, xi=1', rB, 3000, 350))
    rows[-1]['epsilon_for_one_toy_event'] = eps_one

    for d in [275, 350, 370, 375]:
        rC = rate_vector(3000, d, 'Z')
        rows.append(row('C hybrid Z, qZ=1/4, xi=1', rC, 3000, d))

    for d in [270, 600, 800, 850]:
        rD = rate_vector(3000, -d, 'D', eps=1e-6*100/(2*E))
        rows.append(row('D CS exothermic, Gp=1e-6GeV^-2, excited fraction=1', rD, 3000, -d))
        
    # An explicit subdominant excited-population benchmark, with the full
    # propagator retained. First solve the rate's exact epsilon^2 scaling.
    eps_start = 5e-5
    rd0 = rate_vector(3000, -800, 'D', fraction=.1, mvec=1, gdark=.1, eps=eps_start)
    eps_sub = eps_start/np.sqrt(integral(rd0)*EXPOSURE)
    rD_ = rate_vector(3000, -800, 'D', fraction=.1, mvec=1, gdark=.1, eps=eps_sub)
    rows.append(row('D CS, f2=0.1, mV=1GeV, g=0.1, eps fitted to one toy', rD_, 3000, -800))
    rows[-1]['epsilon_for_one_toy_event'] = eps_sub

    # The hybrid benchmark E is the TC state or the DTC replacement in
    # realization I, provided the matching current qZ=1/4 is established.
    rE1 = rate_vector(3000, 275, 'E', mvec=.15, gdark=.03)
    rows.append(row('E hybrid Z+V, node=110keV, xi=1', rE1, 3000, 275))
    rows[-1]['epsilon_node'] = -((131.3-54)*np.sqrt(2)*GF*.25-54*(1-4*SW2)*np.sqrt(2)*GF*.25)*(2*122*110e-6+.15**2)/(3*.03*E*54)
    rE2 = rate_vector(3000, 290, 'E', mvec=.15, gdark=.03, node_kev=140)
    rows.append(row('E hybrid Z+V, node=140keV, xi=1', rE2, 3000, 290))

    output = {'scope': 'recoil-only, approximate isotopes and acceptance; not LZ PLR',
              'halo':{'v0_kms': V0, 'vesc_kms': VESC, 'vEarth_kms': VE, 'rho_GeV_cm3': RHO},
              'isotopes_count_abundances': ISOTOPES,
              'N_target_per_tonne': N_TONNE,
              'efficiency': '0.96 low logistic(5.4, 1.5) high erfc(269.9, 12), true energy',
              'dipole': 'charge response only, magnetic nuclear-spin response omitted',
              'scenerios': rows}
    output['sensitivity_one_at_a_time'] = sensitivity_scan()
    (ROOT(0) / 'compare_scenario.json').write_text(json.dumps(output, indent=2))

    # Area-normalized comparison; no background, detector smearing or veto.
    rA = rate_dipole_charge(1000, 300, 1.8e-3)
    rC = rate_vector(3000, 370, 'Z')
    rD = rate_vector(3000, -800, 'D', mvec=10, eps=1e-6*100/(2*E))
    curves=[('A dipole, 300 keV', rA),
            ('B vector, 350 keV', rB),
            ('C $Z$, 370 keV', rC),
            ('D exothermic, 800 keV', rD),
            ('E hybrid $Z+V$, 290 keV', rE2)]
    
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for lab, r in curves:
        norm = integral(r)
        ax.plot(ER*1e6, r/norm*1e-6, label=lab)
    ax.axvspan(225, 270, color='gray', alpha=.15, label='event vicinity (illustrative)')
    ax.set(
        xlim=(20, 330),
        xlabel='true xenon recoil [keV]',
        ylabel='normalized toy accepted density [keV$^{-1}$]'
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)

    fig.tight_layout()
    fig.savefig(ROOT(0) / 'compare_scenario_shapes.pdf')
    fig.savefig(ROOT(0) / 'compare_scenario_shapes.png', dpi=600)


if __name__ == '__main__':
    main()
