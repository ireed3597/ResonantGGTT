import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (10,8)

import pandas as pd 
import numpy as np
import sys
import common
import json
import scipy.interpolate as spi
import scipy.stats as sps
import scipy.optimize as spo
import os
from optimisation import tools

def getWindowBoundaries(score, N=10, d=1):
  """
  Return a list of boundaries that correspond to slices of N
  events. The events are ordered according to score. The windows
  can overlap. The distance between windows is given by d.
  """

  sorted_score = np.sort(score)[::-1] # highest score is first

  higher_bound_idx = np.arange(0, len(score)-N, d)
  lower_bound_idx = higher_bound_idx + N
  bound_idx = np.array([lower_bound_idx, higher_bound_idx]).T

  bounds = sorted_score[bound_idx]

  assert (bounds > 1).any() == 0
  assert (bounds < 0).any() == 0

  return bounds

def get_bkg_mc_cdf(bkg_mc, score):
  bkg_score = bkg_mc[score].to_numpy()
  bkg_w = bkg_mc["weight"].to_numpy()

  sort_idx = np.argsort(bkg_score)
  bkg_score = np.concatenate([[0], bkg_score[sort_idx], [1]])
  w = bkg_w[sort_idx]
  cdf = np.concatenate([[0], np.cumsum(w), [sum(w)]]) # unnormalised
  spline = spi.interp1d(bkg_score, cdf)
  return spline

def get_fit_cdf(f):
  N = 10000
  cdf_x = np.linspace(0, 1, N)
  cdf = np.cumsum(f(cdf_x)) / N
  spline = spi.interp1d(cdf_x, cdf)

  return spline

def loadDataframe(bkg_mc_is_data=False):
  score_cols = list(filter(lambda x: "score" in x, common.getColumns(sys.argv[1])))
  sig_proc_example = score_cols[0].split("score_")[1]
  other_cols = ["weight", "Diphoton_mass", "process_id"]
  print("Reading")
  df = pd.read_parquet(sys.argv[1], columns=other_cols)
  s = df.process_id >= 0
  df = df[s]
  scores_dfs = [df]
  for score in score_cols:
    scores_dfs.append(pd.read_parquet(sys.argv[1], columns=[score])[s])
  df = pd.concat(scores_dfs, axis=1)
  print("Read")

  del scores_dfs

  pres = tools.get_pres(sig_proc_example)
  df = df[(df.Diphoton_mass > pres[0]) & (df.Diphoton_mass < pres[1])]
  #df = df[df.Diphoton_mass < 100]

  #sidebands = ((m>100)&(m<115)) | ((m>135)&(m<180))
  #df = df[sidebands]

  with open(sys.argv[2], "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  data = df[df.process_id==0]

  if bkg_mc_is_data:
    part_50 = data.sample(frac = 0.5)
    rest_part_50 = data.drop(part_50.index)
    data = part_50
    bkg = rest_part_50
  else:
    bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"]]
    bkg = df[df.process_id.isin(bkg_ids)]
  
  bkg.loc[:, "weight"] *= data.weight.sum() / bkg.weight.sum()

  print(bkg.weight.min())

  return data, bkg

def plotHistogram(N_pred, N, savepath):
  p = sps.poisson(N).pmf

  n, edges = np.histogram(N_pred, range=(0, 50), bins=50, density=True)
  n_unnormed, edges = np.histogram(N_pred, range=(0, 50), bins=50)
  n_unnormed[n_unnormed==0] = 1
  err = n / np.sqrt(n_unnormed)
  err[err==0] = min(err[err!=0])

  chi2 = np.sum((n-p(edges[:-1]))**2 / err**2)
  chi2_dof = chi2 / len(n)
  med, std = np.median(N_pred), np.std(N_pred)
  label = r"Pred. $N_{sidebands}$" + "\nMedian = %.2f\nStd. = %.2f"%(med, std)
  label += "\n" + r"$\chi^2$/d.o.f = %.2f"%chi2_dof
  
  bin_centers = (edges[:-1]+edges[1:])/2
  plt.hist(bin_centers, edges, weights=n, label=label)
  plt.errorbar(bin_centers, n, err, color="k", linestyle="")

  x = np.arange(0, 50, 1)
  plt.plot(x, p(x), label=r"Poisson($\mu$=%d)"%N)
  plt.axvline(x=10, color="r", label="Window size\n"+r"($N_{sidebands}$ in data)")
  plt.legend()
  plt.xlabel(r"$N_{sidebands}$")
  plt.ylim(bottom=0)
  plt.savefig(savepath)
  plt.clf()

  return med, std, chi2_dof

def plotDataVsMC(data, bkg_mc, score, savepath, fitted_f=None):
  sumw_data, edges = np.histogram(data[score], bins=20, range=(0, 1.0))
  error_data = np.sqrt(sumw_data)
  sumw_mc, edges = np.histogram(bkg_mc[score], edges, weights=bkg_mc.weight)
  sumw2_mc, edges = np.histogram(bkg_mc[score], edges, weights=bkg_mc.weight**2)
  error_mc = np.sqrt(sumw2_mc)
  bin_centers = (edges[1:] + edges[:-1])/2

  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

  axs[0].errorbar(bin_centers, sumw_data, error_data, fmt="ko", label="Data")
  axs[0].hist(bin_centers, edges, weights=sumw_mc, label="MC")
  axs[0].fill_between(edges, np.append(sumw_mc-error_mc, 0), np.append(sumw_mc+error_mc, 0), step="post", alpha=0.5, color="grey", zorder=8) #background uncertainty
  axs[0].set_ylim(top=max(sumw_data*1.2))
  axs[0].legend(ncol=2)
  
  if fitted_f is not None:
    x = np.linspace(0, 1, 100)
    axs[0].plot(x, fitted_f(x)/len(sumw_data)) 

  ratio = sumw_data / sumw_mc
  ratio_err = ratio * np.sqrt((error_data/sumw_data)**2 + (error_mc/sumw_mc)**2)

  axs[1].errorbar(bin_centers, ratio, ratio_err, fmt="ko", label="tot. ratio = %.2f"%(sum(sumw_data)/sum(sumw_mc)))
  axs[1].set_ylabel("Data / MC")
  axs[1].axhline(1, color="k")
  axs[1].set_xlabel(score, fontsize=14)
  axs[1].legend()

  plt.savefig(savepath)
  plt.clf()
  plt.close()

def selectScoreRegion(data, bkg_mc, score, range, renorm_bkg_mc=True):
  def select(df, score, range):
    df = df[(df[score] > range[0]) & (df[score] <= range[1])]
    df.loc[:, score] -= range[0]
    df.loc[:, score] *= 1 / (range[1]-range[0])
    df.loc[:, score] /= df[score].max()
    return df

  data = select(data, score, range)
  bkg_mc = select(bkg_mc, score, range)
  if renorm_bkg_mc:
    bkg_mc.loc[:, "weight"] *= data.weight.sum() / bkg_mc.weight.sum()

  return data, bkg_mc

def performFit(score, savepath, weights=None):
  if weights is None:
    weights = np.ones_like(score)

  sumw, edges = np.histogram(score, range=(0, 1), bins=20, weights=weights, density=True)
  sumw_unnormed, edges = np.histogram(score, edges, weights=weights, density=False)
  n, edges = np.histogram(score, edges)
  sumw2, edges = np.histogram(score, edges, weights=(weights*(sumw[0]/sumw_unnormed[0]))**2, density=False)

  def fitFunction(x, a):
    c = 1 - a/2
    return a*x + c

  def fitFunctionCurve(x, a, b):
    c = 1 + (1/a)*(np.exp(a*(1-b))-np.exp(-a*b))
    return -np.exp(a*(x-b))+c

  x = (edges[1:]+edges[:-1]) / 2
  err = np.sqrt(sumw2)

  #plt.scatter(sumw, err)
  #plt.savefig(savepath[:-4]+"_error_vs_sumw")
  #plt.clf()
  #err_per_sqrt_sumw = np.mean(err/np.sqrt(sumw))
  err_per_sqrt_sumw = np.sqrt(sumw2.sum()) / np.sqrt(sumw.sum())
  #error_per_sqrt_n = np.mean(err/np.sqrt(n))
  err = err_per_sqrt_sumw * np.sqrt(sumw)
  #err = error_per_sqrt_n * np.sqrt(n)

  bounds = [[0, 0], [20, 2]]
  popt, pcov = spo.curve_fit(fitFunctionCurve, x, sumw, sigma=err, p0=[10, 1.5], bounds=bounds)
  perr = np.sqrt(np.diag(pcov))
 
  # for i in range(len(popt)):
  #   if np.isclose(popt[i], bounds[0][i]) or np.isclose(popt[i],bounds[1][i]):
  #     print("p%d = %.2f hit a bound [%.2f, %.2f]"%(i, popt[i], bounds[0][i], bounds[1][i]))
  plt.errorbar(x, sumw, err, fmt="ko")
  xx = np.linspace(0, 1, 100)
  plt.plot(xx, fitFunctionCurve(xx, *popt), label=",".join(["%.3f +- %.3f"%(popt[i],perr[i]) for i in range(len(popt))]))
  plt.ylim(bottom=0)
  plt.legend()
  plt.savefig(savepath)
  plt.clf()

  return lambda x: weights.sum()*fitFunctionCurve(x, *popt)


def runTests(data, bkg_mc, savepath, N=10, d=10, do_fit=False):
  scores = list(filter(lambda x: "score" in x, data.columns))
  bkg_mc.loc[:, "weight"] *= data.weight.sum() / bkg_mc.weight.sum()

  metrics = []
  for score in sorted(scores):
    print(score)
    selected_data = data[[score, "Diphoton_mass", "weight"]]
    selected_bkg_mc = bkg_mc[[score, "Diphoton_mass", "weight"]]

    sig_proc = score.split("score_")[1]
    sr = tools.get_sr(sig_proc)
    selected_data = selected_data[(selected_data.Diphoton_mass < sr[0]) | (selected_data.Diphoton_mass > sr[1])]
    selected_bkg_mc = selected_bkg_mc[(selected_bkg_mc.Diphoton_mass < sr[0]) | (selected_bkg_mc.Diphoton_mass > sr[1])]
    selected_bkg_mc.loc[:, "weight"] *= selected_data.weight.sum() / selected_bkg_mc.weight.sum()

    plotDataVsMC(selected_data, selected_bkg_mc, score, os.path.join(savepath, "%s_data_mc.png"%score))

    selected_data, selected_bkg_mc = selectScoreRegion(selected_data, selected_bkg_mc, score, (0.5, 1.0), renorm_bkg_mc=True)
    #selected_data, selected_bkg_mc = selectScoreRegion(data, bkg_mc, score, (0.1, 1.0), renorm_bkg_mc=True)

    if do_fit:
      #fitted_f = performFit(selected_data[score], os.path.join(savepath, "%s_fit.png"%score))
      fitted_f = performFit(selected_bkg_mc[score], os.path.join(savepath, "%s_fit.png"%score), weights=selected_bkg_mc.weight)
    else:
      fitted_f = None

    plotDataVsMC(selected_data, selected_bkg_mc, score, os.path.join(savepath, "%s_data_mc_half.png"%score), fitted_f=fitted_f)
    
    select_range = (0.9, 1.0)
    selected_data, selected_bkg_mc = selectScoreRegion(selected_data, selected_bkg_mc, score, select_range, renorm_bkg_mc=False)
    if do_fit:
      #fitted_f_scaled = lambda x: (select_range[1]-select_range[0])*fitted_f(x*(select_range[1]-select_range[0])+select_range[0])
      fitted_f_scaled = lambda x: (selected_data.weight.sum()/selected_bkg_mc.weight.sum()) * (select_range[1]-select_range[0])*fitted_f(x*(select_range[1]-select_range[0])+select_range[0])
    else:
      fitted_f_scaled = None
    plotDataVsMC(selected_data, selected_bkg_mc, score, os.path.join(savepath, "%s_data_mc_sr.png"%score), fitted_f=fitted_f_scaled)

    data_score = selected_data[score].to_numpy()
    bounds = getWindowBoundaries(data_score, N=N, d=d)

    data_score_list = np.array([data_score]*len(bounds))
    N_in_data = ((data_score_list.T > bounds[:,0]) & (data_score_list.T <= bounds[:,1])).sum(axis=0)
    # in events have same score, we can get different than N events, let's check its not that much
    #assert (sum(N_in_data == N) / len(N_in_data)) > 0.95

    if do_fit:
      bkg_mc_cdf = get_fit_cdf(fitted_f_scaled)
    else:
      bkg_mc_cdf = get_bkg_mc_cdf(selected_bkg_mc, score)

    N_in_mc = bkg_mc_cdf(bounds[:,1]) - bkg_mc_cdf(bounds[:,0])
    metrics.append(plotHistogram(N_in_mc, N, os.path.join(savepath, "%s.png"%score)))

  metrics = np.array(metrics)
  metric_df = pd.DataFrame({"score":scores, "median":metrics[:,0],"std":metrics[:,1], "chi2":metrics[:,2]})
  return metric_df

if __name__=="__main__":
  data, bkg_mc = loadDataframe(bkg_mc_is_data=False)

  #data = data.sample(frac=1.0, replace=True, random_state=2)

  pd.set_option('mode.chained_assignment', None)

  if len(sys.argv) < 4:
    outdir = "misc/compare_optimisation_approaches"
  else:
    outdir = sys.argv[3]

  # bkg_mc_tmp = data.copy()
  # data_tmp = bkg_mc.copy()
  # os.makedirs(f"{outdir}/switch_direction", exist_ok=True)
  # remove_switch_direction = runTests(data_tmp, bkg_mc_tmp, f"{outdir}/switch_direction")

  bkg_mc_tmp = bkg_mc
  os.makedirs(f"{outdir}/remove_do_fit", exist_ok=True)
  remove_do_fit = runTests(data, bkg_mc_tmp, f"{outdir}/remove_do_fit", do_fit=True)

  bkg_mc_tmp = bkg_mc
  os.makedirs(f"{outdir}/remove_nothing", exist_ok=True)
  remove_nothing = runTests(data, bkg_mc_tmp, f"{outdir}/remove_nothing")

  # bkg_mc_tmp = bkg_mc[abs(bkg_mc.weight) < 1]
  # os.makedirs(f"{outdir}/remove_gt_1", exist_ok=True)
  # remove_gt_1 = runTests(data, bkg_mc_tmp, f"{outdir}/remove_gt_1")

  # bkg_mc_tmp = bkg_mc[abs(bkg_mc.weight) < 10*abs(bkg_mc.weight).mean()]
  # os.makedirs(f"{outdir}/remove_gt_10_times", exist_ok=True)
  # remove_gt_10_times = runTests(data, bkg_mc_tmp, f"{outdir}/remove_gt_10_times")

  # bkg_mc_tmp = bkg_mc[abs(bkg_mc.weight) < 100*abs(bkg_mc.weight).mean()]
  # os.makedirs(f"{outdir}/remove_gt_100_times", exist_ok=True)
  # remove_gt_100_times = runTests(data, bkg_mc_tmp, f"{outdir}/remove_gt_100_times")

  # bkg_mc_tmp = bkg_mc[bkg_mc.process_id != 8]
  # os.makedirs(f"{outdir}/remove_gjet", exist_ok=True)
  # remove_gjet = runTests(data, bkg_mc_tmp, f"{outdir}/remove_gjet")

  # bkg_mc_tmp = bkg_mc[bkg_mc.process_id != 5]
  # os.makedirs(f"{outdir}/remove_ttjet", exist_ok=True)
  # remove_ttjet = runTests(data, bkg_mc_tmp, f"{outdir}/remove_ttjet")

  bkg_mc_tmp = bkg_mc[(bkg_mc.process_id != 5) & (bkg_mc.process_id != 8)]
  os.makedirs(f"{outdir}/remove_jets", exist_ok=True)
  remove_jets = runTests(data, bkg_mc_tmp, f"{outdir}/remove_jets")

  #titles = ["remove_nothing", "remove_do_fit", "remove_gt_1", "remove_gt_10_times", "remove_gt_100_times", "remove_gjet", "remove_ttjet", "remove_jets"]
  titles = ["remove_nothing", "remove_do_fit", "remove_jets"]

  for title in titles:
    df = locals()[title]
    plt.hist(df.chi2, range=(0, 10), bins=10, label=title, histtype="step")
  plt.legend()
  plt.savefig(f"{outdir}/chi2_hist.png")
  plt.clf()
  
  chi2_mean, chi2_std = [], []
  better_titles = []
  for title in titles:
    df = locals()[title]
    chi2_mean.append(df.chi2.mean())
    chi2_std.append(df.chi2.std())
    better_titles.append(title.replace("remove_", "").replace("_times", "x"))
  plt.errorbar(better_titles, chi2_mean, chi2_std, fmt="ko")
  print(pd.DataFrame({"title":titles, "mean":chi2_mean, "std":chi2_std}))
  plt.ylabel(r"Average $\chi^2$/d.o.f")
  plt.savefig(f"{outdir}/chi2_means.png")
  plt.clf()


