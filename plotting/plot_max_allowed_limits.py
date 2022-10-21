import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import pandas as pd

limits = pd.DataFrame({"MX":[410,410,410,500,500,500,600,600,600,700,700,700],
                                       "MY":[70,100,200,70,100,200,70,100,200,70,100,200], 
                                       "limit":[4.08,8.85,4.06,0.916,1.62,1.26,0.214,0.365,0.370,0.0580,0.103,0.120]})

for MX in limits.MX.unique():
  limits_plot = limits[limits.MX==MX]
  plt.errorbar(limits_plot.MY, limits_plot.limit, fmt='o-', label=r"$m_X=%d$"%MX)
plt.xlabel(r"$m_Y$")
plt.ylabel(r"maximum $\sigma$ [fb]")
plt.title(r"$ggF\rightarrow A \rightarrow h_{125}(\rightarrow\tau\tau) a(\rightarrow \gamma\gamma)$")
plt.yscale("log")
plt.ylim(top=plt.ylim()[1]*2)
#plt.xlim(right=plt.xlim()[1]+50)
plt.legend(ncol=2)
plt.savefig("max_allowed_limits.pdf")