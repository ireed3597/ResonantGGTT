import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

import scipy.interpolate as spi
import scipy.optimize as spo

from signalModelling.signal_fit import fitDCB
import signalModelling.signal_fit as signal_fit

import argparse
import os
import json
import common

mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

class interp1d(spi.interp1d):
  def __init__(self, x, y, kind='cubic', bounds_error=True):
    if len(x) < 4:
      super().__init__(x, y, kind='linear', bounds_error=bounds_error)
    else:
      super().__init__(x, y, kind, bounds_error=bounds_error)

def fitSignalModel(df, m, outdir):  
  if sum(df.MX==m) < 100:
    #find closest mass with > 100 events
    possible_masses, counts = np.unique(df.MX, return_counts=True)
    possible_masses_100 = possible_masses[counts>=100]
    if len(possible_masses_100) > 0:
      m = possible_masses_100[np.argmin(abs(possible_masses_100-m))]
    else:
      m = possible_masses[np.argmax(counts)]
    print("Using %d as substitute"%m)

  #popt, perr = fitDCB(df[df.MX==m], fit_range=[115,135], savepath=os.path.join(outdir, "mx_%d.png"%m))
  popt, perr = fitDCB(df[df.MX==m], fit_range=[115,135], savepath=None)

  return popt, perr

def doInterpolation(MX, to_fit_masses, norms_nominal, skip_closest=False):
  if skip_closest:
    idx = np.argsort(abs(MX-to_fit_masses))[0]
    if idx == 0: idx = 1 #do not remove the very first entry
    elif idx == len(to_fit_masses)-1: idx = len(to_fit_masses)-2 #do not remove the very last entry

    not_idx = lambda arr, idx: arr[np.arange(len(arr))!=idx]
    m = not_idx(to_fit_masses, idx)
    n = not_idx(norms_nominal, idx)
  else:
    m = to_fit_masses
    n = norms_nominal

  norm_cubic_spline = interp1d(m, n, kind='cubic')
  norm_linear_spline = interp1d(m, n, kind='linear')

  norm_cubic = float(norm_cubic_spline(MX))
  norm_linear = float(norm_linear_spline(MX))

  return norm_cubic, norm_linear

def deriveModels(original_df, proc_dict, optim_results, original_outdir):
  #derive model for every mass point which you can find from optim_results

  all_masses = np.array(sorted([int(entry["sig_proc"].split("M")[1]) for entry in optim_results]))
  nominal_masses = np.array(sorted([int(entry["sig_proc"].split("M")[1]) for entry in optim_results if entry["sig_proc"] in proc_dict.keys()]))

  nSR = len(optim_results[0]["category_boundaries"]) - 1
  models = {str(year):{str(SR):{} for SR in range(nSR)} for year in np.unique(original_df.year)}

  for year in np.unique(original_df.year):
  #for year in [2016]:
    print(year)
    df_year = original_df[original_df.year==year]
    
    for entry in optim_results[:]:
      MX = int(entry["score"].split("_")[-1].split("M")[1])

      closest_masses = nominal_masses[np.argsort(abs(nominal_masses-MX))]
      closest_mass = closest_masses[0]

      above_masses = closest_masses[closest_masses>MX][:3]
      below_masses = closest_masses[closest_masses<MX][:3]
      to_fit_masses = np.concatenate((below_masses, above_masses))
      if entry["sig_proc"] in proc_dict.keys(): #if nominal mass
        to_fit_masses = np.concatenate(([MX], to_fit_masses))
      to_fit_masses = np.sort(to_fit_masses)

      df_year = tagSignals(df_year, entry, proc_dict)

      print(MX)
      print(to_fit_masses)

      for SR in np.unique(df_year.SR):
        print(SR)
        df = df_year[df_year.SR==SR]

        outdir = os.path.join(original_outdir, str(year), str(SR), str(MX))
        os.makedirs(outdir, exist_ok=True)

        popt, perr = fitSignalModel(df, closest_mass, outdir)
        norms_nominal = np.array([df.loc[df.MX==m, "weight"].sum()/common.lumi_table[year] for m in to_fit_masses])
        norms_nominal_err = norms_nominal / np.sqrt([sum(df.MX==m) for m in to_fit_masses])

        #set tiny norms to zero
        if min(norms_nominal) < 0.001:
          norms_nominal = np.zeros_like(norms_nominal)
          norms_nominal_err = np.zeros_like(norms_nominal_err)

        models[str(year)][str(SR)][str(MX)] = {str(to_fit_masses[i]):{"norm": float(norms_nominal[i]), "norm_err": float(norms_nominal_err[i])} for i in range(len(to_fit_masses))}
        
        norm_cubic, norm_linear = doInterpolation(MX, to_fit_masses, norms_nominal)
        norm_cubic_skip, norm_linear_skip = doInterpolation(MX, to_fit_masses, norms_nominal, skip_closest=True)
        
        if (norm_cubic != 0) and (norm_linear != 0):
          norm_cubic_systematic = 1 + abs((norm_cubic-norm_cubic_skip)/norm_cubic)
          norm_linear_systematic = 1 + abs((norm_linear-norm_linear_skip)/norm_linear)
        else:
          norm_cubic_systematic = 1.0
          norm_linear_systematic = 1.0
        
        if norm_cubic_systematic < norm_linear_systematic:
          models[str(year)][str(SR)][str(MX)]["this mass"] = {"norm": norm_cubic, "norm_systematic": norm_cubic_systematic, "norm_spline":"cubic", "parameters":list(popt)}
        else:
          models[str(year)][str(SR)][str(MX)]["this mass"] = {"norm": norm_linear, "norm_systematic": norm_linear_systematic, "norm_spline":"linear", "parameters":list(popt)}

        #check shape assumption
        if closest_mass == MX: #if nominal_mass
          popt_check, perr_check = fitSignalModel(df, closest_masses[1], outdir)
          checkInterpolation(df, MX, popt, popt_check, outdir)


  with open(os.path.join(original_outdir, "model.json"), "w") as f:
    json.dump(models, f, indent=4)

def checkInterpolation(df, m, popt, popt_check, outdir):
  if sum(df.MX==m)<100 : return None

  fit_range = (115, 135)
  nbins=50
  bin_centers, sumw, errors = signal_fit.histogram(df[df.MX==m], fit_range, nbins)

  signal_fit.plotFitComparison(bin_centers, sumw, errors, fit_range, popt, popt_check, os.path.join(outdir, "mx_%d_shape_check.png"%m), normed=True)
  #signal_fit.plotFitComparison(bin_centers, sumw, errors, fit_range, popt_nominal, popt_interp, os.path.join(outdir, "mx_%d_interp_check_normed.png"%m), normed=True)

def tagSignals(df, entry, proc_dict):
  df["SR"] = -1

  boundaries = entry["category_boundaries"][::-1]
  scores = list(filter(lambda x: "intermediate" in x, df.columns))

  for score in scores:
    proc = score.split("intermediate_transformed_score_")[1]

    if proc in proc_dict.keys():
      for i in range(len(boundaries)-1):
        selection = (df[score] <= boundaries[i]) & (df[score] > boundaries[i+1]) & (df.process_id==proc_dict[proc])
        df.loc[selection, "SR"] = i

  return df[df.SR!=-1]

def main(args):
  df = pd.read_parquet(args.parquet_input)
  df = df[df.y==1]
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  common.add_MX_MY(df, proc_dict)
   
  with open(args.optim_results) as f:
     optim_results = json.load(f)
  
  deriveModels(df, proc_dict, optim_results, args.outdir)

  return df

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--step', type=float, default=10.0)
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  df = main(args)