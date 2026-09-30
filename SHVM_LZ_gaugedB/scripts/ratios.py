from rec import *
h=Halo(); E=np.linspace(1,1200,4800)
# single-isotope support check vs ST (98-727 keV at mN=122 GeV)
mA=122.; m=1000.; vm=vmin(E*1e-6,mA,m,3e-4); ok=E[vm<=794.2]; print('support mN=122: %.0f-%.0f keV'%(ok.min(),ok.max()))
def parts(m,d,skin=1.037):
    r=spectrum_O1(m,d,1e-42,h,E,skin=skin)
    roi=np.trapezoid(r*lz_eff(E),E); w=(E>=350)&(E<=650); he=np.trapezoid(r[w],E[w])*0.9
    w2=(E>=270)&(E<=350); gap=np.trapezoid(r[w2],E[w2])
    return roi,gap,he
print('HE-SB/ROI ratio R_HE (Helm, neutron skin), and ROI shape: fraction in 200-270')
for m in (1000.,3000.):
    for d in (250,275,300,320,340,350,360,370):
        roi,gap,he=parts(m,d)
        if roi<=0: continue
        print('  m=%4.0f d=%3d  R_HE=%.3f  R_gap=%.3f'%(m,d,he/roi,gap/roi))
