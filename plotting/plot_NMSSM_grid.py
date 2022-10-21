import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
plt.rcParams['figure.constrained_layout.use'] = True
import numpy as np

def getGrid(path):
  MXs = []
  MYs = []
  with open(path, "r") as f:
    for dataset in f.readlines():
      MXs.append(int(dataset.split("_")[7]))
      MYs.append(int(dataset.split("_")[9]))
  return np.array(MXs), np.array(MYs)

Y_gg = getGrid("plotting/Y_gg_query.txt")
Y_tautau = getGrid("plotting/Y_tautau_query.txt")

plt.scatter(Y_tautau[0], Y_tautau[1], label=r"$Y\rightarrow\tau\tau$ only")
plt.scatter(Y_gg[0], Y_gg[1], label=r"$Y\rightarrow\tau\tau$ and $Y\rightarrow \gamma\gamma$")

plt.xlabel(r"$m_X$")
plt.ylabel(r"$m_Y$")
plt.legend()
plt.savefig("NMSSM_grid.png")
plt.savefig("NMSSM_grid.pdf")
plt.clf()

plt.scatter(Y_gg[0][Y_gg[1]<=125], Y_gg[1][Y_gg[1]<=125], label=r"$Y\rightarrow\tau\tau$ and $Y\rightarrow \gamma\gamma$")

plt.xlabel(r"$m_X$")
plt.ylabel(r"$m_Y$")
plt.legend()
plt.savefig("NMSSM_grid_low_mass.png")
plt.savefig("NMSSM_grid_low_mass.pdf")
plt.clf()

import common
plt.gcf().set_size_inches(12.5, 3)
plt.gca().get_yaxis().set_visible(False)
graviton_procs = common.sig_procs["X_HH"]
mxs = [common.get_MX_MY(proc)[0] for proc in graviton_procs]
plt.scatter(mxs, [1 for each in mxs])
plt.xlabel(r"$m_X$")
plt.savefig("X_HH_grid.png")
plt.savefig("X_HH_grid.pdf")