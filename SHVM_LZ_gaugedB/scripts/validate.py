from rec import *

halo = Halo()
E = np.linspace(1, 1000, 4000)


def N(m, d, s, acc=True, lo=0, hi=1000, skin=1.0):
    r = spectrum_O1(m, d, s, halo, E, skin=skin)
    w = (E >= lo) & (E <= hi)
    r = r*(lz_eff(E) if acc else 1)
    return np.trapezoid(r[w], E[w])*2.84


for skin in (1.0, 1.037):
    r = spectrum_O1(1000, 300, 6.5e-43, halo, E, skin=skin)
    nz = E[r > 0]
    print('skin=%.3f  N_before=%.3f  N_after=%.3f  peak=%.0f keV  support %.0f-%.0f keV  2nd-min=%.0f keV' % (
        skin, N(1000, 300, 6.5e-43, False, skin=skin), N(1000, 300, 6.5e-43, True, skin=skin), E[np.argmax(r)], nz.min(), nz.max(),
        E[(E > 220) & (E < 320)][np.argmin(r[(E > 220) & (E < 320)])]))
    for d, st in [(300, 2.5e-42), (320, 8.5e-42), (340, 5.5e-41), (350, 2.2e-40)]:
        s1 = 1e-42/N(1000, d, 1e-42, True, skin=skin)
        print('delta=%d: sigma(1 accepted event)=%.2e   [ST: %.1e]  ratio %.2f' % (
            d, s1, st, s1/st))
mu = 1000*122/(1122.)
print('E*=%.1f keV  vthr=%.0f km/s' % (mu/122*300, np.sqrt(2*3e-4/mu)*C))