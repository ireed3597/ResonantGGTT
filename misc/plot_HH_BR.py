import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (8, 6)

import numpy as np

import sys
import common
import json

channels = np.array([r"$\gamma\gamma$", r"$\mathrm{ZZ}$", r"$\mathrm{WW}$", r"$\tau\tau$", r"$bb$"])
BR = np.array([2.27e-3, 2.62e-2, 2.14e-1, 6.27e-2, 5.84e-1])

channels = channels[np.argsort(BR)]
BR = BR[np.argsort(BR)]

BR2 = np.log10(2*np.outer(BR, BR))

plt.imshow(BR2, origin="lower")
plt.xticks(np.arange(len(channels)), labels=channels)
plt.yticks(np.arange(len(channels)), labels=channels)
cbar = plt.colorbar()
#cbar.set_label(r"$\log_{10} BR(\mathrm{HH}\rightarrow XX)$")
plt.title(r"$\log_{10} BR(\mathrm{HH}\rightarrow XX)$", fontsize=22)
plt.savefig("HH_BR.pdf")