import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

import scipy.interpolate as spi

from signalModelling.signal_fit import fitDCB
import signalModelling.signal_fit as signal_fit

import argparse
import os
import json
import common
import sys

mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

class interp1d(spi.interp1d):
  def __init__(self, x, y, kind='cubic', bounds_error=True):
    if len(x) < 4:
      super().__init__(x, y, kind='linear', bounds_error=bounds_error)
    else:
      super().__init__(x, y, kind, bounds_error=bounds_error)
  def __call__(self, x, y=None):
    return super().__call__(x)

# class interp2d(spi.interp2d):
#   def __init__(self, x, y, z, kind='cubic', bounds_error=True):
#     if len(x) < 16:
#       super().__init__(x, y, z, kind='linear', bounds_error=bounds_error)
#     else:
#       super().__init__(x, y, z, kind, bounds_error=bounds_error)

class interp2d():
  def __init__(self, x, y, z, kind='cubic', bounds_error=True):
    self.points = np.array([[x[i],y[i]] for i in range(len(x))])
    self.values = np.array(z)
    self.kind = kind

  def __call__(self, x, y):
    if (type(x) != np.ndarray) and (type(y) != np.ndarray):
      return spi.griddata(self.points, self.values, np.array([[x,y]]), method=self.kind)[0]
    elif type(x) != np.ndarray:
      return spi.griddata(self.points, self.values, np.array([[x,y[i]] for i in range(len(y))]), method=self.kind)
    elif type(y) != np.ndarray:
      return spi.griddata(self.points, self.values, np.array([[x[i],y] for i in range(len(x))]), method=self.kind)
    else:
      return spi.griddata(self.points, self.values, np.array([[x[i],y[i]] for i in range(len(y))]), method=self.kind)

def interp(x, y, z, kind='cubic', bounds_error=True):
  if len(np.unique(x)) == 1:
    return interp1d(y, z, kind, bounds_error)
  elif len(np.unique(y)) == 1:
    return interp1d(x, z, kind, bounds_error)
  else:
    return interp2d(x, y, z, kind, bounds_error)

def fitSignalModel(df, mx, outdir, my=125):  
  if sum((df.MX==mx)&(df.MY==my)) < 100:
    #find closest mass with > 100 events
    possible_masses, counts = np.unique(df[df.MY==my].MX, return_counts=True)
    possible_masses_100 = possible_masses[counts>=100]
    if len(possible_masses_100) > 0:
      mx = possible_masses_100[np.argmin(abs(possible_masses_100-mx))]
    else:
      mx = possible_masses[np.argmax(counts)]
    print("Using mx=%d as substitute"%mx)

  popt, perr = fitDCB(df[(df.MX==mx)&(df.MY==my)], fit_range=[my-10*(my/100),my+10*(my/100)], savepath=os.path.join(outdir, "mx_%d_my_%d.png"%(mx,my)))
  #popt, perr = fitDCB(df[(df.MX==mx)&(df.MY==my)], fit_range=[my-10,my+10], savepath=None)

  popt[1] = popt[1] - my #change mean to a delta (dm)

  return popt, perr

def doSkipClosest(MX, MY, to_fit_masses, norms_nominal):  
  unique_mx = np.unique(to_fit_masses[:, 0])
  unique_my = np.unique(to_fit_masses[:, 1])
  argx = interp1d(unique_mx, np.arange(len(unique_mx)))
  argy = lambda x: interp1d(unique_my, np.arange(len(unique_my)))(x) if len(unique_my)>1 else 0

  #find closest points on the grid
  dist = np.sqrt((MX-to_fit_masses[:,0])**2 + (MY-to_fit_masses[:,1])**2)
  dist_grid = np.sqrt((argx(MX) - argx(to_fit_masses[:,0]))**2 + (argy(MY) - argy(to_fit_masses[:,1]))**2)
  idxs = list(np.argsort(dist_grid)[::-1])
  
  #only keep those which are not on edge of grid
  if len(np.unique(to_fit_masses[:,1])) == 1: #if 1d interp:
    is_edge = lambda idx: (to_fit_masses[idx,0] == min(to_fit_masses[:,0])) or (to_fit_masses[idx,0] == max(to_fit_masses[:,0]))
  else:
    #is corner piece in grid
    is_edge = lambda idx: ((to_fit_masses[idx,0] == min(to_fit_masses[:,0])) or (to_fit_masses[idx,0] == max(to_fit_masses[:,0])))   and   ((to_fit_masses[idx,1] == min(to_fit_masses[:,1])) or (to_fit_masses[idx,1] == max(to_fit_masses[:,1])))
  idx = idxs.pop()
  while is_edge(idx):
    idx = idxs.pop()

  not_idx = lambda arr, idx: arr[np.arange(len(arr))!=idx]
  m = not_idx(to_fit_masses, idx)
  n = not_idx(norms_nominal, idx)

  skipped_mass = to_fit_masses[idx]

  return m, n, skipped_mass

def doInterpolation(MX, MY, to_fit_masses, norms_nominal, skip_closest=False):
  if skip_closest:
    m, n, skipped_mass = doSkipClosest(MX, MY, to_fit_masses, norms_nominal)
  else:
    m = to_fit_masses
    n = norms_nominal
    skipped_mass = None

  norm_cubic_spline = interp(m[:,0], m[:,1], n, kind='cubic')
  norm_linear_spline = interp(m[:,0], m[:,1], n, kind='linear')

  norm_cubic = float(norm_cubic_spline(MX, MY))
  norm_linear = float(norm_linear_spline(MX, MY))

  return norm_cubic, norm_linear, skipped_mass

def deriveModels(original_df, proc_dict, optim_results, original_outdir):
  #derive model for every mass point which you can find from optim_results

  all_masses = [common.get_MX_MY(entry["sig_proc"]) for entry in optim_results]
  nominal_masses = np.array([common.get_MX_MY(entry["sig_proc"]) for entry in optim_results if entry["sig_proc"] in proc_dict.keys()])

  nominal_mxs = np.array(list(set(nominal_masses[:,0])))
  nominal_mys = np.array(list(set(nominal_masses[:,1])))

  nSR = len(optim_results[0]["category_boundaries"]) - 1
  models = {str(year):{str(SR):{} for SR in range(nSR)} for year in np.unique(original_df.year)}

  for year in np.unique(original_df.year):
  #for year in [2016]:
    print(year)
    df_year = original_df[original_df.year==year]
    
    for entry in optim_results[:]:
      MX, MY = common.get_MX_MY(entry["sig_proc"])

      closest_mxs = nominal_mxs[np.argsort(abs(nominal_mxs-MX))]
      closest_mx = closest_mxs[0]

      closest_mys = nominal_mys[np.argsort(abs(nominal_mys-MY))]
      closest_my = closest_mys[0]

      above_mx = closest_mxs[closest_mxs>MX][:3]
      below_mx = closest_mxs[closest_mxs<MX][:3]
      above_my = closest_mys[closest_mys>MY][:3]
      below_my = closest_mys[closest_mys<MY][:3]

      to_fit_mxs = np.sort(np.concatenate((below_mx, above_mx)))
      to_fit_mys = np.sort(np.concatenate((below_my, above_my)))
      if closest_mx == MX:
        to_fit_mxs = np.sort(np.concatenate(([MX], to_fit_mxs)))
      if closest_my == MY:
        to_fit_mys = np.sort(np.concatenate(([MY], to_fit_mys)))
      to_fit_masses = [[mx, my] for mx in to_fit_mxs for my in to_fit_mys if tuple([mx, my]) in all_masses] #convenient to leave as list for tagSignals
      #to_fit_masses = list(filter(lambda x: x in all_masses, to_fit_masses))

      df_tagged = tagSignals(df_year, entry, proc_dict, to_fit_masses)

      to_fit_masses = np.array(to_fit_masses)
      print(MX, MY)
      print(to_fit_masses)

      for SR in range(len(entry["category_boundaries"])-1):
      #for SR in [0]:
        print(SR)
        df = df_tagged[df_tagged.SR==SR]

        outdir = os.path.join(original_outdir, str(year), str(SR), "%d_%d"%(MX,MY))
        os.makedirs(outdir, exist_ok=True)

        norms_nominal = np.array([df.loc[(df.MX==mx)&(df.MY==my), "weight"].sum()/common.lumi_table[year] for mx,my in to_fit_masses])
        norms_nominal_err = norms_nominal / np.sqrt([sum((df.MX==mx)&(df.MY==my)) for mx,my in to_fit_masses])

        #set tiny norms to zero
        norm_closest = df.loc[(df.MX==closest_mx)&(df.MY==closest_my), "weight"].sum()/common.lumi_table[year]
        if norm_closest < 0.001:
          popt = np.array([1, closest_my, 2, 1.2, 10, 1.2, 10]) #doesn't matter but give sensible numbers anyway
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)] = {"%d_%d"%tuple(to_fit_masses[i]):{"norm": 0.0, "norm_err": 0.0} for i in range(len(to_fit_masses))}
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"] = {"norm": 0.0, "norm_systematic": 0.0, "norm_spline":"linear", "parameters":list(popt), "skipped_mass":"%d_%d"%(closest_mx,closest_my), "closest_mass":"%d_%d"%(closest_mx,closest_my)}
          continue

        models[str(year)][str(SR)]["%d_%d"%(MX, MY)] = {"%d_%d"%tuple(to_fit_masses[i]):{"norm": float(norms_nominal[i]), "norm_err": float(norms_nominal_err[i])} for i in range(len(to_fit_masses))}

        popt, perr = fitSignalModel(df, closest_mx, outdir, closest_my)    
       
        norm_cubic, norm_linear, skipped_mass = doInterpolation(MX, MY, to_fit_masses, norms_nominal)
        norm_cubic_skip, norm_linear_skip, skipped_mass = doInterpolation(MX, MY, to_fit_masses, norms_nominal, skip_closest=True)
        
        if (norm_cubic != 0) and (norm_linear != 0):
          norm_cubic_systematic = 1 + abs((norm_cubic-norm_cubic_skip)/norm_cubic)
          norm_linear_systematic = 1 + abs((norm_linear-norm_linear_skip)/norm_linear)
        else:
          norm_cubic_systematic = 1.0
          norm_linear_systematic = 1.0
        
        if norm_cubic_systematic < norm_linear_systematic:
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"] = {"norm": norm_cubic, "norm_systematic": norm_cubic_systematic, "norm_spline":"cubic", "parameters":list(popt)}
        else:
          models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"] = {"norm": norm_linear, "norm_systematic": norm_linear_systematic, "norm_spline":"linear", "parameters":list(popt)}
        models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["skipped_mass"] = "%d_%d"%tuple(skipped_mass)
        models[str(year)][str(SR)]["%d_%d"%(MX, MY)]["this mass"]["closest_mass"] = "%d_%d"%(closest_mx, closest_my)

        #check shape assumption
        # if (closest_mx == MX) and (closest_my == MY): #if nominal_mass
        #   popt_check, perr_check = fitSignalModel(df, skipped_mass[0], outdir, skipped_mass[1])
        #   checkInterpolation(df, MX, MY, popt, popt_check, outdir)

  with open(os.path.join(original_outdir, "model.json"), "w") as f:
    json.dump(models, f, indent=4, sort_keys=True)

def checkInterpolation(df, MX, MY, popt, popt_check, outdir):
  if sum((df.MX==MX)&(df.MY==MY))<100 : return None

  #fit_range = (115, 135)
  fit_range = (MY-10, MY+10)
  nbins=50
  bin_centers, sumw, errors = signal_fit.histogram(df[(df.MX==MX)&(df.MY==MY)], fit_range, nbins)

  popt_c = popt.copy()
  popt_check_c = popt_check.copy()
  popt_c[1] += MY
  popt_check_c[1] += MY

  signal_fit.plotFitComparison(bin_centers, sumw, errors, fit_range, popt_c, popt_check_c, os.path.join(outdir, "mx_%d_my_%d_shape_check.png"%(MX,MY)), normed=True)
  #signal_fit.plotFitComparison(bin_centers, sumw, errors, fit_range, popt_nominal, popt_interp, os.path.join(outdir, "mx_%d_interp_check_normed.png"%m), normed=True)

def tagSignals(df, entry, proc_dict, to_fit_masses):
  df.loc[:, "SR"] = -1

  boundaries = entry["category_boundaries"][::-1]
  scores = list(filter(lambda x: "intermediate" in x, df.columns))

  for score in scores:
    proc = score.split("intermediate_transformed_score_")[1]
    if list(common.get_MX_MY(proc)) in to_fit_masses:

      if proc in proc_dict.keys():
        for i in range(len(boundaries)-1):
          selection = (df[score] <= boundaries[i]) & (df[score] > boundaries[i+1]) & (df.process_id==proc_dict[proc])
          df.loc[selection, "SR"] = i

  return df[df.SR!=-1]

def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=True)
    return True

  df = pd.read_parquet(args.parquet_input)
  df = df[df.y==1]
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  common.add_MX_MY(df, proc_dict)
   
  with open(args.optim_results) as f:
     optim_results = json.load(f)
  
  deriveModels(df, proc_dict, optim_results, args.outdir)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--step', type=float, default=10.0)
  parser.add_argument('--batch', action="store_true")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)