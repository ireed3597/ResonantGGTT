import numpy as np
import scipy.optimize as spo

def createHistograms(data, mc, features, feature_ranges="auto"):
  hists = []
  for i, feature in enumerate(features):
    print(feature)

    if feature_ranges == "auto":
      feature_range = [min([data[feature].quantile(0.0), mc[feature].quantile(0.0)]), max([data[feature].quantile(0.95), mc[feature].quantile(0.95)])]
    else:
      feature_range = feature_ranges[i]
    print(feature_range)

    data_hist, edges = np.histogram(data[feature], bins=100, range=feature_range, weights=data.weight_central)
    data_hist_err = np.sqrt(data_hist)

    mc_hists = []
    mc_hists_err = []
    for proc in mc.bkg_group.unique():
      print(proc)
      N, edges = np.histogram(mc.loc[mc.bkg_group==proc, feature], bins=edges)
      mc_hist, edges = np.histogram(mc.loc[mc.bkg_group==proc, feature], bins=edges, weights=mc.loc[mc.bkg_group==proc, "weight_central"])
      mc_hist_err = np.nan_to_num(mc_hist / np.sqrt(N))

      mc_hists.append(mc_hist)
      mc_hists_err.append(mc_hist_err)

    mc_hists = np.array(mc_hists)
    mc_hists_err = np.array(mc_hists_err)

    hists.append([data_hist, data_hist_err, mc_hists, mc_hists_err])
  return hists

def calculateChi2(k_factors, hists):
  k_factors = np.array(k_factors)
  chi2 = 0
  for i, hist_collection in enumerate(hists):
    mc_hist_sum = np.sum(hist_collection[2] * k_factors[:,np.newaxis], axis=0)
    mc_hist_err_sum = np.sqrt(np.sum((hist_collection[3] * k_factors[:,np.newaxis])**2, axis=0))

    data_hist, data_hist_err = hist_collection[0:2]

    chi2 += np.sum( (data_hist - mc_hist_sum)**2/(data_hist_err**2 + mc_hist_err_sum**2) )
  return chi2

def deriveScaleFactors(data, mc, features, k_factor_bounds=None):
  n_bkg_groups = len(mc.bkg_group.unique())
  p0 = np.ones(n_bkg_groups)

  if k_factor_bounds == None:
    k_factor_bounds = [[0, 5] for i in range(n_bkg_groups)]

  hists = createHistograms(data, mc, features)
  print(calculateChi2(p0, hists))
  res = spo.minimize(calculateChi2, p0, args=(hists), bounds=k_factor_bounds)
  return res.x

def applyScaleFactors(mc, k_factors):
  for i, group in enumerate(mc.bkg_group.unique()):
    mc.loc[mc.bkg_group==group, "weight_central"] *= k_factors[i]

if __name__=="__main__":
  import sys
  import pandas as pd
  import json
  df = pd.read_parquet(sys.argv[1])

  with open(sys.argv[2]) as f:
    proc_dict = json.load(f)['sample_id_map']
  print(proc_dict)

  bkg_groups = {
  'Diphoton': ["Diphoton_MGG-40to80", "Diphoton_MGG-80toInf"],
  'GJets': ['GJets_HT-100To200', 'GJets_HT-200To400', 'GJets_HT-400To600', 'GJets_HT-40To100', 'GJets_HT-600ToInf'],
  #'TT': ['TTGG', 'TTGamma', 'TTJets'],
  #'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
  #'VGamma': ['WGamma', 'ZGamma'],
  'QCD': ["QCD_Pt-30To40_MGG-80toInf", "QCD_Pt-30ToInf_MGG-40to80", "QCD_Pt-40ToInf_MGG-80toInf"]
  }

  df["bkg_group"] = ""
  for group in bkg_groups.keys():
    for proc in bkg_groups[group]:
      df.loc[df.process_id==proc_dict[proc], "bkg_group"] = group

  for proc in proc_dict.keys():
    if proc_dict[proc] in np.unique(df.loc[df.bkg_group=="", "process_id"]):
      print(proc)

  data = df[df.process_id == proc_dict["Data"]]
  mc = df[df.bkg_group != ""]

  print(data.weight_central.sum())
  print(mc.weight_central.sum())

  #bounds = [[0.99, 1.01], [0.1, 10], [0.1, 10], [0.1,10], [0.1, 10]]
  bounds = [[0.1, 5], [0.1, 5], [0.1, 5]]
  k_factors = deriveScaleFactors(data, mc, ["Diphoton_pt", "LeadPhoton_pt", "SubleadPhoton_pt"], bounds)
  print(k_factors)

  applyScaleFactors(mc, k_factors)
  print(data.weight_central.sum())
  print(mc.weight_central.sum())

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.hist(data.Diphoton_pt, bins=50, range=(0, 200), weights=data.weight_central, histtype='step')
  plt.hist(mc.Diphoton_pt, bins=50, range=(0, 200), weights=mc.weight_central, histtype='step')
  plt.savefig("test_kfactors.png")