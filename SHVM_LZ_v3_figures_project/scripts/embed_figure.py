"""Build a single TeX source with the scenario comparison plot as PGFPlots data."""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import SHVM_LZ_v3_figures_project.scripts.compare_scenarios as scenario
import numpy as np

ROOT = lambda x : Path(__file__).resolve().parents[x]
source = (ROOT(1) / "SHVM_LZ_v3_revised.tex").read_text()
source = source.replace("\\usepackage[colorlinks", "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\usepackage[colorlinks", 1)

labels_and_rates = [
    (r"A dipole, $300\,\mathrm{keV}$", scenario.rate_dipole_charge(1000, 300, 1.8e-3)),
    (r"B vector, $350\,\mathrm{keV}$", scenario.rate_vector(3000, 350, "B", eps=1e-4)),
    (r"C $Z$, $370\,\mathrm{keV}$", scenario.rate_vector(3000, 370, "Z")),
    (r"D exothermic, $800\,\mathrm{keV}$", scenario.rate_vector(3000, -800, "D", mvec=10,
                                                 eps=1e-6 * 100 / (2 * scenario.E))),
    (r"E hybrid $Z+V$, $290\,\mathrm{keV}$", scenario.rate_vector(3000, 290, "E", mvec=.15,
                                                    gdark=.03, node_kev=140)),
]
colors = ["blue!75!black", "orange!90!black", "green!55!black", "red!75!black", "violet"]
plot = [r"\begin{figure}[t]", r"\centering", r"\begin{tikzpicture}",
        r"\begin{axis}[width=.96\linewidth,height=7.4cm,xmin=20,xmax=300,ymin=0,ymax=.029,",
        r" xlabel={true xenon recoil [keV]},",
        r" ylabel={normalized toy accepted density [keV$^{-1}$]},",
        r" grid=major,grid style={gray!20},",
        r" legend style={font=\scriptsize,fill=white,draw=gray!25,at={(.02,.98)},anchor=north west,legend columns=2}]",
        r"\addplot[draw=none,fill=gray!15,forget plot] coordinates {(225,0) (270,0) (270,.029) (225,.029)}\closedcycle;"]
for color, (label, rate) in zip(colors, labels_and_rates):
    normalized = rate / scenario.integral(rate) * 1e-6
    indices = np.flatnonzero((scenario.ER * 1e6 >= 20) & (scenario.ER * 1e6 <= 300))[::4]
    indices = np.unique(np.r_[indices, np.argmax(normalized)])
    plot.append(r"\addplot+[color=" + color + r",thick,no marks] coordinates {")
    for start in range(0, len(indices), 6):
        plot.append(" ".join(f"({scenario.ER[j]*1e6:.2f},{normalized[j]:.8g})" for j in indices[start:start+6]))
    plot.append("};")
    plot.append(r"\addlegendentry{" + label + "}")
plot.extend([r"\end{axis}", r"\end{tikzpicture}",
            r"\caption{Area-normalized accepted recoil surrogates. The grey",
            r"band is illustrative; the observed event has substantial energy",
            r"uncertainty. A has $m=1\TeV,\delta=300\keV$; B has",
            r"$m=3\TeV,\delta=350\keV$; C has $m=3\TeV,\delta=370\keV$;",
            r"D has $m=3\TeV,\delta=-800\keV$; E has",
            r"$m=3\TeV,\delta=290\keV$ and an amplitude node at",
            r"$140\keV$. The curves omit nuclear magnetic responses,",
            r"secondary deposits and detector resolution.}",
            r"\label{fig:shapes}", r"\end{figure}"])
start = source.index(r"\begin{figure}[t]")
end = source.index(r"\end{figure}", start) + len(r"\end{figure}")
source = source[:start] + "\n".join(plot) + source[end:]
(ROOT(1) / "SHVM_LZ_v3_figures_embedded.tex").write_text(source)
