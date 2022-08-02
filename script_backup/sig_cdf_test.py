import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

import scipy.optimize as spo

def tan(x, a, b, c, d, f):
  e = (1/(1-f))*np.arctan(c/d)
  #a = (e*d*np.cos(b*(x-f))**2) / (b*np.cos(e*(x-f))**2)
  return (a*np.tan((x-f)*b) - c)*(x<f) + (d*np.tan((x-f)*e) - c)*(x>=f)

def tan(x, b, c, d, f):
  e = (1/(1-f))*np.arctan(c/d)
  a = (e*d)/b
  return (a*np.tan((x-f)*b) - c)*(x<f) + (d*np.tan((x-f)*e) - c)*(x>=f)

df = pd.read_parquet("training_output/NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100/output.parquet")
sig = df[df.process_id == 3]
#df = pd.read_parquet("paramBDTTest/radionM500_HHggTauTau/output.parquet")
#print(np.unique(df.process_id))
#sig = df[df.process_id == 32]

print(sig.head())

sig.sort_values("intermediate_transformed_score", inplace=True)
#cdf = np.log10(np.cumsum(sig.weight_central)/np.sum(sig.weight_central)).to_numpy()
cdf = np.log10(np.cumsum(sig.weight_central)/np.sum(sig.weight_central)).to_numpy()

plt.plot(sig.intermediate_transformed_score, cdf)

#hist, edges = np.histogram(cdf, bins=50, range=(0, 1))
#cdf = hist
#score = (edges[:1]+edges[1:])/2

score = np.linspace(0, 1, 100)
cdf = [cdf[np.argmin(np.abs(sig.intermediate_transformed_score-s))] for s in score]

#popt, pcov = spo.curve_fit(tan, sig.transformed_score[:100], cdf[:100], p0=[1, np.pi-0.2, -3, 1, np.pi-0.2])
popt, pcov = spo.curve_fit(tan, score, cdf, p0=[np.pi-0.2, -3, 1, 0.5])

print(list(popt))

#popt = [0.9, np.pi-0.2, -3]

plt.plot(score, tan(score, *popt), label="Fit")
plt.xlabel("Score")
plt.ylabel("log10(Signal CDF)")
plt.legend()
plt.savefig("generic_sig_cdf_fit.png")