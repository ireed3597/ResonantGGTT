import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

def getGrid(path):
  MXs = []
  MYs = []
  with open(path, "r") as f:
    for dataset in f.readlines():
      MXs.append(int(dataset.split("_")[7]))
      MYs.append(int(dataset.split("_")[9]))
  return MXs, MYs

Y_gg = getGrid("plotting/Y_gg_query.txt")
Y_tautau = getGrid("plotting/Y_tautau_query.txt")

plt.scatter(Y_tautau[0], Y_tautau[1], label=r"$Y\rightarrow\tau\tau$ only")
plt.scatter(Y_gg[0], Y_gg[1], label=r"$Y\rightarrow\tau\tau$ and $Y\rightarrow \gamma\gamma$")


plt.xlabel(r"$m_X$")
plt.ylabel(r"$m_Y$")
plt.legend()
plt.savefig("NMSSM_grid.png")
plt.savefig("NMSSM_grid.pdf")