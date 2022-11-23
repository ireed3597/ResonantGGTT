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

class DCBFit:
  def __init__(self, sumw):
    self.sumw = sumw #sum of all events

  def __call__(self, x, mean, sigma, beta1, m1, beta2, m2):
    N = 1
    y_pred = dcb(x, N, mean, sigma, beta1, m1, beta2, m2)
    y_pred *= self.sumw / sum(y_pred)
    return y_pred

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

def dcb_gaus(x, N1, N2, mean, sigma, beta1, m1, beta2, m2):
  beta1, m1, beta2, m2 = np.abs(beta1), np.abs(m1), np.abs(beta2), np.abs(m2)

  with np.errstate(all='ignore'):
    A1 = np.power(m1/beta1, m1) * np.exp(-beta1**2/2)
    B1 = m1/beta1 - beta1
    A2 = np.power(m2/beta2, m2) * np.exp(-beta2**2/2)
    B2 = m2/beta2 - beta2

    xs = (x-mean)/sigma

    middle = N1*np.exp(-xs**2/2)*(-beta1 < xs)*(xs < beta2)
    left = np.nan_to_num(N1*A1*np.power(B1-xs, -m1)*(xs<=-beta1), nan=0.0)
    right = np.nan_to_num(N1*A2*np.power(B2+xs, -m2)*(xs>=beta2), nan=0.0)

  dcb = left + middle + right
  gaus = N2*np.exp(-xs**2/2)
  
  return dcb + gaus

def getChi2(popt, x, y, y_err, func):
  y_pred = func(x, *popt)
  y_pred *= sum(y) / (sum(y_pred) + 1e-8)

  chi2 = np.sum(np.power((y-y_pred)/y_err, 2)) / len(x)
  return chi2

def chi2Fit(x, y, p0, bounds, errors, deviate=False, level=0):
  p0_copy = p0.copy()
  if deviate: 
    for j in [3, 4, 5, 6]:
      p0[j] = bounds[0][j] + np.random.random() * (bounds[1][j]-bounds[0][j])

  popt, pcov = spo.curve_fit(DCBFit(sum(y)), x, y, p0=p0[1:], sigma=errors, bounds=(bounds[0][1:], bounds[1][1:]), ftol=1e-4, xtol=1e-4)
  popt = np.abs(popt)
  perr = np.sqrt(np.diag(pcov))
  popt = np.concatenate(([p0[0]], popt))
  perr = np.concatenate(([0.], perr))
  chi2 = getChi2(popt, x, y, errors, dcb)

  return popt, perr, chi2

def histogram(df, fit_range, nbins):
  sumw, edges = np.histogram(df.Diphoton_mass, bins=nbins, range=fit_range, density=False, weights=df.weight)
  sumw2, edges = np.histogram(df.Diphoton_mass, bins=nbins, range=fit_range, density=False, weights=df.weight**2)
  
  bin_centers = (edges[:-1] + edges[1:])/2
  errors = np.sqrt(sumw2)

  non_zero_indicies = np.arange(len(errors))[errors>0]
  for i, lt in enumerate(errors<=0):
    if lt:
      closest_match = non_zero_indicies[np.argmin(np.abs(non_zero_indicies-i))]
      errors[i] = errors[closest_match]

  sumw[sumw<0] = 0

  return bin_centers, sumw, errors

def plotFit(bin_centers, sumw, errors, fit_range, popt, chi2, savepath):
  plt.errorbar(bin_centers, sumw, errors, fmt="k.", capsize=2)
  x = np.linspace(fit_range[0], fit_range[1], 200)
  #plt.plot(x, dcb(x, *popt))

  popt_normed = popt.copy()
  popt_normed[0] *= sum(sumw) / spi.quad(dcb, fit_range[0], fit_range[1], args=tuple(popt_normed), epsrel=0.001)[0]
  popt_normed[0] *= (fit_range[1]-fit_range[0])/len(sumw)
  plt.plot(x, dcb(x, *popt_normed))

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

def plotFitComparison(bin_centers, sumw, errors, fit_range, popt_nom, popt_interp, savepath):
  plt.errorbar(bin_centers, sumw, errors, fmt="k.", capsize=2)
  x = np.linspace(fit_range[0], fit_range[1], 200)

  popt_nom_normed = popt_nom.copy()
  popt_nom_normed[0] *= sum(sumw) / spi.quad(dcb, fit_range[0], fit_range[1], args=tuple(popt_nom_normed), epsrel=0.001)[0]
  popt_nom_normed[0] *= (fit_range[1]-fit_range[0])/len(sumw)
  chi2_nom = getChi2(popt_nom, bin_centers, sumw, errors, dcb)
  plt.plot(x, dcb(x, *popt_nom_normed), label="MC fit\n" + r"$\chi^2 / dof$=%.2f"%chi2_nom)

  popt_interp_normed = popt_interp.copy()
  popt_interp_normed[0] *= sum(sumw) / spi.quad(dcb, fit_range[0], fit_range[1], args=tuple(popt_interp_normed), epsrel=0.001)[0]
  popt_interp_normed[0] *= (fit_range[1]-fit_range[0])/len(sumw)
  chi2_interp = getChi2(popt_interp, bin_centers, sumw, errors, dcb)
  plt.plot(x, dcb(x, *popt_interp_normed), label="Interpolated fit\n" + r"$\chi^2 / dof$=%.2f"%chi2_interp)

  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.legend()
  plt.savefig(savepath)
  plt.clf()

def fitDCB(df, fit_range, savepath=None, p0=None):
  my = sum(fit_range) / 2

  mean = my
  width = my / 100
  
  nbins = 50

  bin_centers, sumw, errors = histogram(df, fit_range, nbins)

  if p0 is None:
    #N0 = df.weight.sum() / (width*np.sqrt(2*np.pi))
    #N0 = N0 * ((fit_range[1]-fit_range[0]) / nbins)
    N0 = max(sumw)
    p0 = [N0, mean, width, 1.5, 5, 1.5, 15]

  lbounds = [0,       my-width, width*0.5, 0.1, 0.1, 0.1, 0.1]
  hbounds = [p0[0]*2, my+width, width*1.5, 4,   15,  4,   20]
  bounds = (lbounds, hbounds)

  for i in range(len(p0)):
    assert (p0[i]>=lbounds[i]) & (p0[i]<=hbounds[i]), print(i, p0, lbounds, hbounds)

  popt, perr, chi2 = chi2Fit(bin_centers, sumw, p0, bounds, errors)
  
  assert not (popt[:-1] == np.array(lbounds[:-1])).any(), print("Hit lbounds %s"%str(popt))
  assert not (popt[:-1] == np.array(hbounds[:-1])).any(), print("Hit hbounds %s"%str(popt))
  
  if savepath != None:
    plotFit(bin_centers, sumw, errors, fit_range, popt, chi2, savepath)


  return popt, perr

