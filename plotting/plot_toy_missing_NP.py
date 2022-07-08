import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (10,8)

import numpy as np
import scipy.stats as sps

x = np.linspace(250, 550, 500)
y = sps.norm.pdf(x, loc=400, scale=100) 
y = y / max(y)

plt.plot(x, y, label=r"Signal $m_X=400$ GeV")
plt.xlabel(r"Trained $m_X$")
plt.ylabel("Sensitivity")

h = sps.norm.pdf(300, loc=400, scale=100) / sps.norm.pdf(400, loc=400, scale=100)
#plt.plot([300, 300], [min(y), h], 'k--')
#plt.plot([500, 500], [min(y), h], 'k--')
plt.plot([250, 550], [h, h], 'k--')

h = sps.norm.pdf(375, loc=400, scale=100) / sps.norm.pdf(400, loc=400, scale=100)
#plt.plot([375, 375], [min(y), h], 'r--')
#plt.plot([425, 425], [min(y), h], 'r--')
plt.plot([250, 550], [h, h], 'r--')

x = np.arange(275,575,50)
plt.scatter(x, sps.norm.pdf(x, loc=400, scale=100) / sps.norm.pdf(400, loc=400, scale=100), facecolors="r", label="Fine MC", zorder=10)
h = sps.norm.pdf(300, loc=400, scale=100) / sps.norm.pdf(400, loc=400, scale=100)
plt.scatter([300, 500], [h, h], facecolors="k",  label="Coarse MC", zorder=10)

plt.ylim(min(y), 1.2)
plt.legend(ncol=2)

plt.savefig("toy_missing_np.pdf")

