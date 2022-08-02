import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (10,8)

import numpy as np
import sys
import os
import json

years = [2016, 2017, 2018]
cats = [0, 1, 2, 3, 4, 5, 6, 7]
nominal_masses = [260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000]

def loadSystematics(year, SR):
  with open(os.path.join(sys.argv[1], str(year), str(SR), "systematics.json")) as f:
    systematics = json.load(f)
  return systematics

def loadModel(year, SR):
  with open(os.path.join(sys.argv[1], str(year), str(SR), "model.json")) as f:
    model = json.load(f)
  return model

def plotInterpUncert(mx, interp_uncert, savename):
  print(mx, interp_uncert)
  plt.scatter(mx[interp_uncert!=1], interp_uncert[interp_uncert!=1])
  plt.xlabel(r"$m_X$")
  plt.ylabel("Sig. eff. interpolation systematic")
  plt.savefig(os.path.join(sys.argv[1], savename))
  plt.clf()

def plotSigEff(mx, sig_eff, sig_eff_error, interp_uncert, savename, SR):
  s = interp_uncert!=1
  l = sig_eff[s] - (1-interp_uncert[s])*sig_eff[s]
  h = sig_eff[s] + (1-interp_uncert[s])*sig_eff[s]

  plt.scatter(mx, sig_eff, marker='.', label="Intermediate masses")
  #plt.scatter(nominal_masses, sig_eff[np.isin(mx, nominal_masses)], marker='.', label="Nominal masses")
  plt.errorbar(nominal_masses, sig_eff[np.isin(mx, nominal_masses)], sig_eff_error[np.isin(mx, nominal_masses)], fmt='r.', label="Nominal masses")
  plt.fill_between(mx[s], l, h, label="Interpolation Uncert.", alpha=0.2, color="green")
  plt.xlabel(r"$m_X$")
  plt.ylabel("Signal efficiency")
  plt.title("Category %d"%SR)
  plt.legend()
  plt.savefig(os.path.join(sys.argv[1], savename))
  plt.clf()


systematics = loadSystematics(2016, 0)
mx = np.array(sorted([float(m) for m in systematics.keys()]))

inter_uncert = []
for year in years:
  year_uncert = []
  for SR in cats:
    systematics = loadSystematics(year, SR)
    year_uncert.append([systematics["%d.0"%m]["interpolation"] for m in mx])
  inter_uncert.append(year_uncert)

sig_eff = []
sig_eff_error = []
for year in years:
  year_eff = []
  year_eff_error = []
  for SR in cats:
    model = loadModel(year, SR)
    year_eff.append([model["%d.0"%m][0] for m in mx])
    year_eff_error.append([model["%d.0"%m][2] for m in mx])
  sig_eff.append(year_eff)
  sig_eff_error.append(year_eff_error)

inter_uncert = np.array(inter_uncert)
inter_uncert_year_avg = inter_uncert.mean(axis=0)

sig_eff = np.array(sig_eff)
sig_eff_year_avg = sig_eff.mean(axis=0)

sig_eff_error = np.array(sig_eff_error)
sig_eff_error_year_avg = sig_eff_error.mean(axis=0)

for i, SR in enumerate(cats):
  plotInterpUncert(mx, inter_uncert_year_avg[i,:], "interpolation_uncertainty_cat%d"%SR)
  plotSigEff(mx, sig_eff_year_avg[i,:], sig_eff_error_year_avg[i,:], inter_uncert_year_avg[i,:],  "sig_eff_cat%d.pdf"%SR, SR)

  for j, year in enumerate(years):
   plotInterpUncert(mx, inter_uncert[j,i,:], "interpolation_uncertainty_%d_cat%d"%(year, SR))
   plotSigEff(mx, sig_eff[j,i,:], sig_eff_error[j,i,:], inter_uncert[j,i,:],  "sig_eff_%d_cat%d.pdf"%(year, SR), SR)

