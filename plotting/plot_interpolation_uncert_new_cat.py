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

from signalModelling.interpolate_new_cat import interp1d

def plotInterpUncert(mx, interp_uncert, savename):
  print(mx, interp_uncert)
  plt.scatter(mx[interp_uncert!=1], interp_uncert[interp_uncert!=1])
  plt.xlabel(r"$m_X$")
  plt.ylabel("Sig. eff. interpolation systematic")
  plt.savefig(os.path.join(sys.argv[1], savename))
  plt.clf()

def plotSigEff(mx, norms, norm_errs, this_mass, this_mass_norm, this_mass_norm_systematic, savename, SR, spline_kind):
  norms = norms[np.argsort(mx)]
  mx = mx[np.argsort(mx)]

  #spline_all = interp1d(mx[np.argsort(abs(mx-this_mass))[:4]], norms[np.argsort(abs(mx-this_mass))[:4]], bounds_error=False)

  not_idx = lambda arr, idx: arr[np.arange(len(arr))!=idx]
  idx = np.argsort(abs(this_mass-mx))[0]
  if idx == 0: idx = 1 #do not remove the very first entry
  elif idx == len(mx)-1: idx = len(mx)-2 #do not remove the very last entry
  spline_skip = interp1d(not_idx(mx, idx), not_idx(norms, idx), kind=spline_kind)

  x = np.linspace(min(mx), max(mx), 100)
  all_points_line = plt.plot(x, interp1d(mx, norms, kind='cubic')(x), label="All points")[0]
  skip_points_line = plt.plot(x, interp1d(not_idx(mx, idx), not_idx(norms, idx), kind='cubic')(x), label="Skip closest")[0]

  plt.plot(x, interp1d(mx, norms, kind='linear')(x), linestyle='dashed', color=all_points_line.get_color())
  plt.plot(x, interp1d(not_idx(mx, idx), not_idx(norms, idx), kind='linear')(x), linestyle='dashed', color=skip_points_line.get_color())

  #plt.scatter(mx, norms, marker='o', label="Nominal masses")
  plt.errorbar(mx, norms, norm_errs, color=all_points_line.get_color(), fmt='o', capsize=5, label="Nominal masses")
  plt.errorbar([this_mass], [this_mass_norm], [this_mass_norm*(1-this_mass_norm_systematic)], marker='o', capsize=5, label="Target mass", zorder=10)

  plt.xlabel(r"$m_X$")
  plt.ylabel("Signal efficiency")
  plt.title(r"Target $m_X=%d$ Category %d %s spline"%(this_mass, SR, spline_kind))
  plt.legend()
  plt.savefig(savename)
  plt.clf()

with open(sys.argv[1], "r") as f:
  models = json.load(f)

years = list(models.keys())
cats = list(models[years[0]].keys())
masses = list(models[years[0]][cats[0]].keys())

print(years)
print(cats)
print(masses)

nominal_masses = np.array([260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000])
sys_masses = np.array([270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900])

for year in years:
  for cat in cats:
    for mass in masses:
      mx = list(filter(lambda x: "this mass" not in x, models[year][cat][mass].keys()))
      norms = [models[year][cat][mass][m]["norm"] for m in mx]
      norm_errs = [models[year][cat][mass][m]["norm_err"] for m in mx]
      mx = np.array(mx, dtype=int)
      norms = np.array(norms, dtype=float)
      norm_errs = np.array(norm_errs, dtype=float)

      this_mass = int(mass)
      this_mass_norm = models[year][cat][mass]["this mass"]["norm"]
      this_mass_norm_systematic = models[year][cat][mass]["this mass"]["norm_systematic"]
      spline_kind = models[year][cat][mass]["this mass"]["norm_spline"]

      save_dir = os.path.join(sys.argv[2], str(year), str(cat))
      os.makedirs(save_dir, exist_ok=True)
      
      plotSigEff(mx, norms, norm_errs, this_mass, this_mass_norm, this_mass_norm_systematic, os.path.join(save_dir, str(mass)+".png"), int(cat), spline_kind)


      
      