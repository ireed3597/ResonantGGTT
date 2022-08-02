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

from signalModelling.interpolate import interp

def plotInterpUncert(mx, interp_uncert, savename):
  print(mx, interp_uncert)
  plt.scatter(mx[interp_uncert!=1], interp_uncert[interp_uncert!=1])
  plt.xlabel(r"$m_X$")
  plt.ylabel("Sig. eff. interpolation systematic")
  plt.savefig(os.path.join(sys.argv[1], savename))
  plt.clf()

def plot(spline, spline_skip, spline_linear, spline_skip_linear, slice, norms, norm_errs, m, this_mass_norm, this_mass_norm_systematic, savename, SR, spline_kind):
  norms = norms[np.argsort(slice)]
  slice = slice[np.argsort(slice)]

  x = np.linspace(min(slice), max(slice), 100)

  all_points_line = plt.plot(x, spline(x), label="All points")[0]
  skip_points_line = plt.plot(x, spline_skip(x), label="Skip closest")[0]

  plt.plot(x, spline_linear(x), linestyle='dashed', color=all_points_line.get_color())
  plt.plot(x, spline_skip_linear(x), linestyle='dashed', color=skip_points_line.get_color())

  #plt.scatter(mx, norms, marker='o', label="Nominal masses")
  plt.errorbar(slice, norms, norm_errs, color=all_points_line.get_color(), fmt='o', capsize=5, label="Nominal masses")
  plt.errorbar([m], [this_mass_norm], [this_mass_norm*(1-this_mass_norm_systematic)], marker='o', capsize=5, label="Target mass", zorder=10)

  plt.xlabel(r"$m$")
  plt.ylabel("Signal efficiency")
  plt.title(r"Target $m=%d$ Category %d %s spline"%(m, SR, spline_kind))
  plt.legend()
  plt.savefig(savename+".png")
  plt.clf()

def plotSigEff(masses_in_interp, skipped_mass, norms, norm_errs, mx, my, this_mass_norm, this_mass_norm_systematic, savename, SR, spline_kind):
  #finding where the skipped mass is
  not_idx = lambda arr, idx: arr[np.arange(len(arr))!=idx]
  #print(mx, my)
  #print(masses_in_interp)
  #print(skipped_mass)
  idx = np.where((masses_in_interp == skipped_mass).sum(axis=1)==2)[0][0]
  
  spline = interp(masses_in_interp[:,0], masses_in_interp[:,1], norms)
  spline_skip = interp(not_idx(masses_in_interp[:,0],idx), not_idx(masses_in_interp[:,1],idx), not_idx(norms,idx))

  spline_linear = interp(masses_in_interp[:,0], masses_in_interp[:,1], norms, kind='linear')
  spline_skip_linear = interp(not_idx(masses_in_interp[:,0],idx), not_idx(masses_in_interp[:,1],idx), not_idx(norms,idx), kind='linear')

  #plot in slices of mx and my
  mx_slice = masses_in_interp[masses_in_interp[:,1]==my, 0]
  norms_mx_slice = norms[masses_in_interp[:,1]==my]
  norm_errs_mx_slice = norm_errs[masses_in_interp[:,1]==my]
  spline_mx_slice = lambda x: spline(x, my)
  spline_skip_mx_slice = lambda x: spline_skip(x, my)
  spline_linear_mx_slice = lambda x: spline_linear(x, my)
  spline_skip_linear_mx_slice = lambda x: spline_skip_linear(x, my)
  
  my_slice = masses_in_interp[masses_in_interp[:,0]==mx, 1]
  norms_my_slice = norms[masses_in_interp[:,0]==mx]
  norm_errs_my_slice = norm_errs[masses_in_interp[:,0]==mx]
  spline_my_slice = lambda x: spline(mx, x)
  spline_skip_my_slice = lambda x: spline_skip(mx, x)
  spline_linear_my_slice = lambda x: spline_linear(mx, x)
  spline_skip_linear_my_slice = lambda x: spline_skip_linear(mx, x)
  
  if len(mx_slice) > 1:
    plot(spline_mx_slice, spline_skip_mx_slice, spline_linear_mx_slice, spline_skip_linear_mx_slice, mx_slice, norms_mx_slice, norm_errs_mx_slice, mx, this_mass_norm, this_mass_norm_systematic, savename+"_mx_slice", SR, spline_kind)
  if len(my_slice) > 1:
    plot(spline_my_slice, spline_skip_my_slice, spline_linear_my_slice, spline_skip_linear_my_slice, my_slice, norms_my_slice, norm_errs_my_slice, my, this_mass_norm, this_mass_norm_systematic, savename+"_my_slice", SR, spline_kind)


with open(sys.argv[1], "r") as f:
  models = json.load(f)

years = list(models.keys())
cats = list(models[years[0]].keys())
masses = list(models[years[0]][cats[0]].keys())

print(years)
print(cats)
print(masses)

#nominal_masses = np.array([260,270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900,1000])
#sys_masses = np.array([270,280,290,300,320,350,400,450,500,550,600,650,700,750,800,900])

for year in years:
  print(year)
  for cat in cats:
    print(cat)
    for mass in masses:
      if models[year][cat][mass]["this mass"]["closest_mass"] == mass: continue
      if models[year][cat][mass]["this mass"]["norm"] == 0.0:          continue
      print(mass)

      mx, my = mass.split("_")
      mx = int(mx)
      my = int(my)

      masses_in_interp = []
      for each in models[year][cat][mass].keys():
        if each != "this mass":
          masses_in_interp.append([int(each.split("_")[0]), int(each.split("_")[1])])
      masses_in_interp = np.array(masses_in_interp)

      skipped_mass = models[year][cat][mass]["this mass"]["skipped_mass"]
      skipped_mass = np.array(skipped_mass.split("_"), dtype=int)

      norms = [models[year][cat][mass][m]["norm"] for m in models[year][cat][mass].keys() if m != "this mass"]
      norm_errs = [models[year][cat][mass][m]["norm_err"] for m in models[year][cat][mass].keys() if m != "this mass"]
      masses_in_interp = np.array(masses_in_interp, dtype=int)
      norms = np.array(norms, dtype=float)
      norm_errs = np.array(norm_errs, dtype=float)

      this_mass_norm = models[year][cat][mass]["this mass"]["norm"]
      this_mass_norm_systematic = models[year][cat][mass]["this mass"]["norm_systematic"]
      spline_kind = models[year][cat][mass]["this mass"]["norm_spline"]

      save_dir = os.path.join(sys.argv[2], str(year), str(cat))
      os.makedirs(save_dir, exist_ok=True)
      
      plotSigEff(masses_in_interp, skipped_mass, norms, norm_errs, mx, my, this_mass_norm, this_mass_norm_systematic, os.path.join(save_dir, str(mass)), int(cat), spline_kind)


      
      