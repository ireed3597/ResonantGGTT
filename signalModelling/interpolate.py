import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

import scipy.interpolate as spi
import scipy.optimize as spo

from signalModelling.signal_fit import fitDCB

mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

def getSignalNormalisation(df, mass, boundaries):
  masses = sorted(np.unique(df.mX))
  sig_norm = [df.loc[(df.mX==m)&(df.transformed_score>boundaries[0])&(df.transformed_score<=boundaries[1]), "weight_central"].sum()/(59*1000) for m in masses]
  #sig_norm = [df.loc[(df.mX==m), "weight_central"].sum()/(59*1000) for m in masses]
  f = spi.interp1d(masses, sig_norm)
  return float(f(mass))

def getSignalModel(df, mass, boundaries):
  nominal_masses = sorted(np.unique(df.mX))
  #masses.remove(400)
  select = lambda df, m, b: df[(df.mX==m)&(df.transformed_score>b[0])&(df.transformed_score<=b[1])]
  
  popts = []
  perrs = []
  for m in nominal_masses:
    print(len(select(df, m, boundaries)))
    print(len(df[df.mX==m]))
    popt, perr = fitDCB(select(df, m, boundaries), fit_range="auto", plot=True, name="_%d"%m)
    popts.append(popt)
    perrs.append(perr)
  popts = np.array(popts)
  perrs = np.array(perrs)

  n_params = len(popts[0])
  splines = [spi.interp1d(nominal_masses, popts[:, i], kind='cubic') for i in range(n_params)]
  
  try:
    interpolated_parameters = np.array([[splines[i](m) for i in range(n_params)] for m in mass])
  except:
    interpolated_parameters = [splines[i](m) for i in range(n_params)]
  return interpolated_parameters

def deriveRadionModels(bounds):
  mx = [300,400,500,800,1000]
  proc_dict = {18:300, 19:400, 20:500, 21:800, 22:1000}
  dfs = []
  for key in proc_dict.keys():
    df_mx = pd.read_parquet("training_output/Radion_paramBDT/radionM%d_HHggTauTau/output.parquet"%proc_dict[key])
    dfs.append(df_mx[df_mx.process_id==key])

  df = pd.concat([d[d.y==1] for d in dfs])

  df["mX"] = 0
  for proc_id in proc_dict.keys():
    df.loc[df.process_id==proc_id, "mX"] = proc_dict[proc_id]

  norms = [getSignalNormalisation(df, m, bounds) for m in [300, 400, 500, 800, 1000]]
  parameters = getSignalModel(df, [300, 400, 500, 800, 1000], bounds)

  plotInterpolation(parameters)

  models = {}
  for i, m in enumerate(mx):
    models[m] = [norms[i], parameters[i].tolist()]
  
  import json
  with open("model_cat0.json", "w") as f:
    json.dump(models, f, indent=4)

def plotInterpolation(parameters):
  mx = [300,400,500,800,1000]
  for i in range(len(parameters[0])):
    plt.scatter(mx, parameters[:, i])
    plt.savefig("p%d.png"%i)
    plt.clf()

deriveRadionModels([0.95, 1.0])


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