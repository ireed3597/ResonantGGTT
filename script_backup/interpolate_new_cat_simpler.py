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

def getParamSplines(nominal_masses, popts):
  n_params = len(popts[0])
  splines = [spi.interp1d(nominal_masses, popts[:, i], kind='linear') for i in range(n_params)]
  return splines

def fitSignalModels(df, nominal_masses, outdir):  
  popts = []
  perrs = []
  already_fitted_masses = []

  for m in nominal_masses:
    if sum(df.MX==m) < 100:
      #find closest mass with > 100 events
      possible_masses, counts = np.unique(df.MX, return_counts=True)
      possible_masses_100 = possible_masses[counts>=100]
      if len(possible_masses_100) > 0:
        m = possible_masses_100[np.argsort(abs(possible_masses_100-m))[0]]
      else:
        m = possible_masses[np.argmax(counts)]
      print("Using %d as substitute"%m)

    if m in already_fitted_masses:
      idx = already_fitted_masses.index(m)
      popt, perr = popts[idx].copy(), perrs[idx].copy()
    else:
      #popt, perr = fitDCB(df[df.MX==m], fit_range=[115,135], savepath=os.path.join(outdir, "mx_%d.png"%m))
      popt, perr = fitDCB(df[df.MX==m], fit_range=[115,135], savepath=None)

    popts.append(popt)
    perrs.append(perr)
  popts = np.array(popts)
  perrs = np.array(perrs)

  return popts, perrs


def deriveModels(original_df, nominal_masses, masses, original_outdir):
  for year in np.unique(original_df.year):
    print(year)
    for SR in np.unique(original_df.SR):
      print(SR)
      df = original_df[(original_df.year==year)&(original_df.SR==SR)]

      outdir = os.path.join(original_outdir, str(year), str(SR))
      os.makedirs(outdir, exist_ok=True)

      norms_nominal = np.array([df.loc[(df.MX==m), "weight"].sum()/common.lumi_table[year] for m in nominal_masses])
      N = np.array([sum(df.MX==m) for m in nominal_masses])
      norms_nominal_errors = norms_nominal / np.sqrt(N)
      norm_spline = interp1d(nominal_masses, norms_nominal, kind='cubic')
      norms = np.array([float(norm_spline(m)) for m in masses])
      
      print("Fitting signal models")     
      popts, perrs = fitSignalModels(df, nominal_masses, outdir)
      splines = getParamSplines(nominal_masses, popts)  
      parameters = np.array([[splines[i](m) for i in range(len(popts[0]))] for m in masses])

      print("Plotting interpolation")
      popts_n = np.concatenate([popts, norms_nominal[:,np.newaxis]], axis=1)
      #perrs_n  = np.concatenate([perrs, np.zeros_like(norms_nominal)[:,np.newaxis]], axis=1)
      perrs_n  = np.concatenate([perrs, norms_nominal_errors[:,np.newaxis]], axis=1)
      parameters_n = np.concatenate([parameters, norms[:,np.newaxis]], axis=1)
      plotInterpolation(nominal_masses, masses, popts_n, perrs_n, parameters_n, outdir)

      print("Dumping models")
      models = {}
      for i, m in enumerate(masses):
        if m in nominal_masses:
          models[m] = [norms[i], parameters[i].tolist(), norms_nominal_errors[np.where(nominal_masses==m)[0][0]]]
        else:
          models[m] = [norms[i], parameters[i].tolist(), 0.0]

      with open(os.path.join(outdir, "model.json"), "w") as f:
        json.dump(models, f, indent=4)

      print("Checking interpolation")
      for m in nominal_masses[1:-1]:
       checkInterpolation(df, nominal_masses, m, popts, outdir)

def plotInterpolation(nominal_masses, masses, popts, perrs, parameters, outdir):
  parameter_names = [r"$N$", r"$\bar{m}_{\gamma\gamma}$", r"$\sigma$", r"$\beta_l$", r"$m_l$", r"$\beta_r$", r"$m_r$", "Signal Efficiency"]
  for i in range(len(parameters[0])):
    plt.scatter(masses, parameters[:, i], marker=".", label="Intermediate masses")
    plt.errorbar(nominal_masses, popts[:, i], perrs[:, i], fmt='ro', label="Nominal masses")
    plt.legend()
    plt.ylabel(parameter_names[i])
    plt.xlabel(r"$m_X$")
    plt.savefig(os.path.join(outdir, "p%d.png"%i))
    plt.clf()

def checkInterpolation(df, nominal_masses, m, popts, outdir):
  idx = np.where(nominal_masses==m)[0][0]
  idx1 = idx - 1
  idx2 = idx + 1

  fit_range = (115, 135)
  nbins=50
  bin_centers, sumw, errors = signal_fit.histogram(df[df.MX==m], fit_range, nbins)

  splines = getParamSplines(nominal_masses[[idx1, idx2]], popts[[idx1, idx2]])

  popt_interp = [spline(m) for spline in splines]
  popt_nominal = popts[idx]
  signal_fit.plotFitComparison(bin_centers, sumw, errors, fit_range, popt_nominal, popt_interp, os.path.join(outdir, "mx_%d_interp_check.png"%m))
  signal_fit.plotFitComparison(bin_centers, sumw, errors, fit_range, popt_nominal, popt_interp, os.path.join(outdir, "mx_%d_interp_check_normed.png"%m), normed=True)

def tagSignals(df, optim_dir, proc_dict):
  df["SR"] = -1

  with open(os.path.join(optim_dir, "optim_results.json"), "r") as f:
    optim_results = json.load(f)

  for proc in proc_dict.keys():
    if proc_dict[proc] in np.unique(df.process_id):
      score_name = "intermediate_transformed_score_%s"%proc

      for entry in optim_results:
        if score_name == entry["score"]:
          boundaries = entry["category_boundaries"][::-1]
          break

      for i in range(len(boundaries)-1):
        selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1]) & (df.process_id == proc_dict[proc])
        df.loc[selection, "SR"] = i

  return df[df.SR!=-1]

def main(args):
  df = pd.read_parquet(args.parquet_input)
  df = df[df.y==1]
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  common.add_MX_MY(df, proc_dict)
  print(np.unique(df.MX))

  masses = np.arange(df.MX.min(), df.MX.max()+args.step, args.step)
  print(masses)

  tagSignals(df, args.optim_results, proc_dict)

  nominal_masses = np.sort(np.unique(df.MX))
  deriveModels(df, nominal_masses, masses, args.outdir)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--step', type=float, default=10.0)
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)


"""
mx = [300,400,500,800,1000]
proc_dict = {30:300, 31:400, 32:500, 33:800, 34:1000}
dfs = []
for key in proc_dict.keys():
  #dfs.append(pd.read_parquet("training_output/radionM%d_HHggTauTau_low_mass/output.parquet"%m))
  df_mx = pd.read_parquet("paramBDTTest/radionM%d_HHggTauTau/output.parquet"%proc_dict[key])
  dfs.append(df_mx[df_mx.process_id==key])

df = pd.concat([d[d.y==1] for d in dfs])

df.loc[:, "weight_central"] *= 2

df["mX"] = 0
for proc_id in proc_dict.keys():
  df.loc[df.process_id==proc_id, "mX"] = proc_dict[proc_id]

# for bound in [0.5, 0.8, 0.9, 0.95, 0.99]:
#   x = np.linspace(300, 1000, 50)
#   plt.scatter(x, [getSignalNormalisation(df, xi, [bound,0.9]) for xi in x], label="%.2f < score <= 0.9"%bound)
#   plt.scatter(mx, [getSignalNormalisation(df, xi, [bound,0.9]) for xi in mx])
#   plt.legend()
# plt.savefig("test_normalisation.png")
# plt.clf()

for bounds in [[0, 0.84], [0.84, 0.95], [0.95, 1]]:
  x = np.linspace(300, 1000, 50)
  plt.scatter(x, [getSignalNormalisation(df, xi, bounds) for xi in x], label=str(bounds))
  plt.scatter(mx, [getSignalNormalisation(df, xi, bounds) for xi in mx])
  plt.legend()
plt.savefig("test_normalisation.png")
plt.clf()

select = lambda df, m, b: df[(df.mX==m)&(df.transformed_score>b[0])&(df.transformed_score<=b[1])]
popts = []
perrs = []
for m in mx:
  popt, perr = fitDCB(select(df, m, [0.84, 0.95]), plot=True, fit_range=(110,140), name="_%d"%m)
  popts.append(popt)
  perrs.append(perr)
popts = np.array(popts)
perrs = np.array(perrs)

params = []
x = np.linspace(300, 1000, 71)
for xi in x:
  print(xi)
  popt = getSignalModel(df, xi, [0.84, 0.95])
  params.append(popt)
params = np.array(params)

print(params.shape)

def initialGuess(popts, i):
  #a = (popts[-1,i]-popts[0,i])/700
  a = 0
  b = popts[-1,i]
  if popts[-1,i] < popts[0,i]:
    c = 1
  else:
    c = -1
  #c = 1
  d = -1
  e = 400
  f = 400
  return [a, b, c, d, e, f]

for i in range(len(params[0])):
  plt.scatter(x, params[:,i])
  plt.errorbar(mx, popts[:, i], perrs[:, i], fmt="ro")

  x_fine = np.linspace(300, 1000, 200)
  #poly = lambda x, a, b, c, d, e, f : a*x + b + c/(x-e) + 0*d/(x-f)
  #res = spo.curve_fit(poly, mx, popts[:, i], initialGuess(popts, i), sigma=perrs[:, i], bounds=([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, 0, 400, 400]))
  #print(res[0])
  #plt.plot(x_fine, poly(x_fine, *res[0]))
    
  plt.plot()
  plt.savefig("p%d.png"%i)
  plt.clf()
"""