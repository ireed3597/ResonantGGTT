import pandas as pd
import scipy.stats as sps
import scipy.optimize as spo
import scipy.integrate as spi
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

from scipy.special import factorial

mplhep.set_style("CMS")

def dcb(x, N, mean, sigma, beta1, m1, beta2, m2):
  beta1, m1, beta2, m2 = np.abs(beta1), np.abs(m1), np.abs(beta2), np.abs(m2)

  with np.errstate(all='ignore'):
    A1 = np.power(m1/beta1, m1) * np.exp(-beta1**2/2)
    B1 = m1/beta1 - beta1
    A2 = np.power(m2/beta2, m2) * np.exp(-beta2**2/2)
    B2 = m2/beta2 - beta2

    xs = (x-mean)/sigma

    middle = N*np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
    left = np.nan_to_num(N*A1*np.power(B1-xs, -m1)*(xs<=-beta1), nan=0.0)
    right = np.nan_to_num(N*A2*np.power(B2+xs, -m2)*(xs>=beta2), nan=0.0)

  return left + middle + right

# def dcb(x, N, mean, sigma, beta1, m1, beta2, m2):
#   beta1, m1, beta2, m2 = np.abs(beta1), np.abs(m1), np.abs(beta2), np.abs(m2)

#   #A1 = np.power(m1/beta1, m1) * np.exp(-beta1**2/2)
#   B1 = m1/beta1 - beta1
#   #A2 = np.power(m2/beta2, m2) * np.exp(-beta2**2/2)
#   B2 = m2/beta2 - beta2

#   xs = (x-mean)/sigma

#   middle = N*np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
  
#   left = np.nan_to_num( N * np.power((m1/beta1)/(B1-xs), m1) * np.exp(-beta1**2/2) * (xs<=-beta1) )
#   right = np.nan_to_num( N * np.power((m2/beta2)/(B2+xs), m2) * np.exp(-beta2**2/2) * (xs>=beta2) )

#   return left + middle + right


def chi2Fit(x, y, p0, errors, deviate=False):
  lbounds = [0, 120, 0.5, 0.5, 0.01, 0.5, 0.01]
  hbounds = [5, 130, 2, 4, 10, 4, 10]

  bounds = (lbounds, hbounds)

  p0_copy = p0.copy()
  if deviate: p0 += np.random.normal(scale=0.1, size=len(p0))

  try:
    popt, pcov = spo.curve_fit(dcb, x, y, p0=p0, sigma=errors, bounds=bounds)
    popt = np.abs(popt)
    perr = np.sqrt(np.diag(pcov))
  except Exception as e:
    print(e)
    #popt, perr, chi2 = chi2Fit(x, y, p0_copy, errors, deviate=True)
    popt, perr = p0, np.zeros_like(p0)

  chi2 = np.sum(np.power((y-dcb(x, *popt))/errors, 2)) / len(x)

  return popt, perr, chi2

log_factorial = lambda n: n*np.log(n) - n + 0.5*np.log(2*np.pi*n)
# log_factorial = lambda n: n*np.log(n) - n

def NLL(k, l):
  return k*np.log(l) - l - log_factorial(k)

# def NLLFit(x, y, p0):
#   res = spo.minimize()

def histogram(df, fit_range, nbins):
  sumw, edges = np.histogram(df.Diphoton_mass, bins=nbins, range=fit_range, density=False, weights=df.weight)
  N, edges = np.histogram(df.Diphoton_mass, bins=nbins, range=fit_range, density=False)
  bin_centers = (edges[:-1] + edges[1:])/2

  with np.errstate(all='ignore'): errors = sumw / np.sqrt(N)
  errors = np.nan_to_num(errors)
  non_zero_indicies = np.arange(len(errors))[errors>0]
  for i, lt in enumerate(errors<=0):
    if lt:
      closest_match = non_zero_indicies[np.argmin(np.abs(non_zero_indicies-i))]
      errors[i] = errors[closest_match]

  return bin_centers, sumw, errors

def plotFit(bin_centers, sumw, errors, fit_range, popt, chi2, savepath):
  plt.errorbar(bin_centers, sumw, errors, fmt="k.", capsize=2)
  x = np.linspace(fit_range[0], fit_range[1], 200)
  plt.plot(x, dcb(x, *popt))
  plt.xlabel(r"$m_{\gamma\gamma}$")
  parameter_names = [r"$\bar{m}_{\gamma\gamma}$", r"$\sigma$", r"$\beta_l$", r"$m_l$", r"$\beta_r$", r"$m_r$"]
  text = "DCB Fit"
  for i, name in enumerate(parameter_names):
    text += "\n" + name + r"$=%.2f$"%popt[i+1]
  #text = "DCB Fit" "\n" r"$\bar{m}_{\gamma\gamma}=%.2f$" "\n" r"$\sigma=%.2f$" "\n" r"$\beta_l=%.2f$" "\n" r"$m_l=%.2f$" "\n" r"$\beta_r=%.2f$" "\n" r"$m_r=%.2f$"%tuple(popt[1:])
  plt.text(min(x), max(sumw+errors), text, verticalalignment='top')
  plt.text(max(x), max(sumw+errors), r"$\chi^2 / dof$=%.2f"%chi2, verticalalignment='top', horizontalalignment='right')
  plt.savefig(savepath)
  plt.clf()

def plotFitComparison(bin_centers, sumw, errors, fit_range, popt_nominal, popt_interp, savepath, normed=False):
  plt.errorbar(bin_centers, sumw, errors, fmt="k.", capsize=2)
  x = np.linspace(fit_range[0], fit_range[1], 200)
  plt.plot(x, dcb(x, *popt_nominal), label="Nominal fit")

  popt_interp_copy = popt_interp.copy()
  if normed: popt_interp_copy[0] *= spi.quad(dcb, fit_range[0], fit_range[1], args=tuple(popt_nominal), epsrel=0.001)[0] / spi.quad(dcb, fit_range[0], fit_range[1], args=tuple(popt_interp_copy), epsrel=0.001)[0]
  plt.plot(x, dcb(x, *popt_interp_copy), label="Interpolated fit")
  plt.xlabel(r"$m_{\gamma\gamma}$")

  plt.legend()
  plt.savefig(savepath)
  plt.clf()

def fitDCB(df, fit_range="auto", savepath=None, p0=None):
  mean = df.Diphoton_mass.mean()
  width = df.Diphoton_mass.quantile(0.5+0.67/2) - df.Diphoton_mass.quantile(0.5) #approx gauss width
  if fit_range == "auto":
    fit_range = [df.Diphoton_mass.mean()-3*width, df.Diphoton_mass.mean()+3*width] #fit in +-4 sigma range

  nbins = 50
  N0 = df.weight.sum() / (width*np.sqrt(2*np.pi))
  N0 = N0 * ((fit_range[1]-fit_range[0]) / nbins)

  bin_centers, sumw, errors = histogram(df, fit_range, nbins)

  if p0 is None: p0 = [N0, mean, width, 1, 1, 1, 1]
  popt, perr, chi2 = chi2Fit(bin_centers, sumw, p0, errors)

  if savepath != None:
    plotFit(bin_centers, sumw, errors, fit_range, popt, chi2, savepath)

  return popt, perr

