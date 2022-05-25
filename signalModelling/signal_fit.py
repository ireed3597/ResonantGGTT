import pandas as pd
import scipy.stats as sps
import scipy.optimize as spo
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

def dcb(x, N, mean, sigma, beta1, m1, beta2, m2):
  A1 = np.power(m1/np.abs(beta1), m1) * np.exp(-beta1**2/2)
  B1 = m1/np.abs(beta1) - np.abs(beta1)
  A2 = np.power(m2/np.abs(beta2), m2) * np.exp(-beta2**2/2)
  B2 = m2/np.abs(beta2) - np.abs(beta2)

  xs = (x-mean)/sigma

  with np.errstate(invalid='ignore'):
    left = np.nan_to_num(N*A1*np.power(B1-xs, -m1)*(xs<=-beta1))
    middle = N*np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
    right = np.nan_to_num(N*A2*np.power(B2+xs, -m2)*(xs>=beta2))

  return left + middle + right

def fitDCB(df, fit_range="auto", plot=False, name=""):
  if fit_range == "auto":
    width = df.Diphoton_mass.quantile(0.5+0.67/2) - df.Diphoton_mass.quantile(0.5)
    print(width)
    fit_range = [df.Diphoton_mass.mean() - 3*width, df.Diphoton_mass.mean()+3*width]
    #fit_range = [df.Diphoton_mass.mean() - 2.5*df.Diphoton_mass.std(), df.Diphoton_mass.mean()+2.5*df.Diphoton_mass.std()]
    #fit_range = [df.Diphoton_mass.quantile(0.025), df.Diphoton_mass.quantile(0.995)]

  sumw, edges = np.histogram(df.Diphoton_mass, bins=100, range=fit_range, density=False, weights=df.weight_central)
  N, edges = np.histogram(df.Diphoton_mass, bins=100, range=fit_range, density=False)
  bin_centers = (edges[:-1] + edges[1:])/2

  N0 = df.weight_central.sum()/np.sqrt(2*np.pi)
  
  errors = sumw / np.sqrt(N)
  errors[errors==0] = errors[errors>0].min()
  #errors[errors==0] = 1

  popt, pcov = spo.curve_fit(dcb, bin_centers, sumw, p0=[N0, (fit_range[1]+fit_range[0])/2, 1, 1, 2, 1, 2], sigma=errors)
  perr = np.sqrt(np.diag(pcov))

  if plot:
    plt.errorbar(bin_centers, sumw, errors, fmt="k.", capsize=2)
    x = np.linspace(fit_range[0], fit_range[1], 200)
    plt.plot(x, dcb(x, *popt))
    plt.savefig("test_fit%s.png"%name)
    plt.clf()

  return popt, perr

