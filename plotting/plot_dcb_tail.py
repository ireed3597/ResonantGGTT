import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)
#plt.rcParams['figure.constrained_layout.use'] = True
import numpy as np

from signalModelling.signal_fit import dcb

popt = [1, 124.81, 1.39, 1.29, 3.82, 1.69]
xlow = popt[1]+popt[2]*popt[3]
xhigh = 137.5
m = np.linspace(xlow, xhigh, 100)

fig, ax = plt.subplots(figsize=(12.5, 10))

ax.plot(m, dcb(m, *popt, 1), label=r"$m_r=1$")
#ax.plot(m, dcb(m, *popt, 10), label=r"$m_r=10$")
ax.plot(m, dcb(m, *popt, 20), label=r"$m_r=20$")
ax.plot(m, dcb(m, *popt, 50), label=r"$m_r=50$")
#ax.plot(m, dcb(m, *popt, 100), label=r"$m_r=100$")
ax.plot(m, dcb(m, *popt, 150), label=r"$m_r=150$")
ax.set_ylim(bottom=-0.01)
ax.legend(loc="upper left", frameon=True)
ax.set_xlabel(r"$m_{\gamma\gamma}$")
#plt.yscale("log")

#axins = ax.inset_axes([132, 0.2, 4, 0.2])
axins = ax.inset_axes([0.4, 0.4, 0.58, 0.58])
axins.plot(m, dcb(m, *popt, 1), label=r"$m_r=1$")
#axins.plot(m, dcb(m, *popt, 10), label=r"$m_r=10$")
axins.plot(m, dcb(m, *popt, 20), label=r"$m_r=20$")
axins.plot(m, dcb(m, *popt, 50), label=r"$m_r=50$")
#axins.plot(m, dcb(m, *popt, 100), label=r"$m_r=100$")
axins.plot(m, dcb(m, *popt, 150), label=r"$m_r=150$")
axins.set_xlim(128,131)
axins.set_ylim(0, 0.08)
axins.set_xticklabels([])
axins.set_yticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="black")

plt.savefig("tail_comparison.png")

from scipy.integrate import quad

res_20 = quad(dcb, xlow, xhigh, args=(*popt, 20))[0]
res_150 = quad(dcb, xlow, xhigh, args=(*popt, 150))[0]
print(res_20, res_150)
print((res_20-res_150)/res_20)

res_20 = quad(dcb, 112.5, 137.5, args=(*popt, 20))[0]
res_150 = quad(dcb,112.5, 137.5, args=(*popt, 150))[0]
print(res_20, res_150)
print((res_20-res_150)/res_20)

res_20 = quad(dcb, 112.5, 137.5, args=(1, 124.81, 1.39, 1.18, 3.82, 1.69, 20))[0]
res_150 = quad(dcb,112.5, 137.5, args=(1, 124.81, 1.39, 1.28, 3.82, 1.69, 150))[0]
print(res_20, res_150)
print((res_20-res_150)/res_20)