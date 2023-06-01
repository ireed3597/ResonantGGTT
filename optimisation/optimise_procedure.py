import pandas as pd
import numpy as np
import argparse
import os
import json
import common
import sys
import copy

from optimisation.limit import getBoundariesPerformance
from optimisation.tools import get_pres,get_sr
from numba import jit

import scipy.optimize as spo
import scipy.interpolate as spi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pickle

def getProcId(proc_dict, proc):
  if proc in proc_dict.keys():
    return proc_dict[proc]
  else:
    print(f'Warning: the process "{proc}" is not in the proc_dict')
    return None

def getGJetIds(proc_dict):
  gjet_ids = [getProcId(proc_dict, proc) for proc in common.bkg_procs["GJets"]] + [getProcId(proc_dict, "TTJets")] + [getProcId(proc_dict, "DiPhoton_Low")]
  #gjet_ids = [getProcId(proc_dict, proc) for proc in common.bkg_procs["GJets"]] + [getProcId(proc_dict, "DiPhoton_Low")]
  gjet_ids = [each for each in gjet_ids if each != None]
  return gjet_ids

def loadDataFrame(args):
  columns = common.getColumns(args.parquet_input)
  columns_load = ["Diphoton_mass", "weight", "year", "process_id"]
  columns_load += list(filter(lambda x: args.score in x, columns))

  df = pd.read_parquet(args.parquet_input, columns=columns_load)

  m = df.Diphoton_mass
  pres = get_pres(args.sig_procs[0]) #is same for all signal processes in one analysis
  df = df[((m >= pres[0]) & (m <= pres[1]))]  
  
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  data = df[df.process_id == proc_dict["Data"]]
  sigs = {sig_proc: df[df.process_id == proc_dict[sig_proc]] for sig_proc in args.sig_procs}

  if not args.mc_as_bkg:
    bkg = data # use data as background
  else:
    bkg_ids = [getProcId(proc_dict, proc) for proc in common.bkg_procs["all"]]
    gjet_ids = getGJetIds(proc_dict)
    #chosen_ids = set(bkg_ids).difference(gjet_ids)
    chosen_ids = bkg_ids
    bkg = df[df.process_id.isin(chosen_ids)]
    bkg.loc[:, "weight"] *= data.weight.sum() / bkg.weight.sum()
  
  if args.bootstrap_i != 0:
    rng = np.random.default_rng(seed=args.bootstrap_i)
    bkg = bkg.loc[rng.choice(bkg.index, size=len(bkg), replace=True)]

  return bkg, sigs, proc_dict

def getBoundaries(bkg, nbkg, score_name="score", savepath=None):
  """
  Return boundaries which correspond to events found at indexes given by nbkg.
  The intention is that the bkg dataframe is already sorted by score such that the boundaries
  correspond to selecting the nbkg[0]-highest scoring events, followed by the next nbkg[1]...
  """
  score = bkg[score_name].to_numpy()
  #print(score[:20])
  boundaries = [0] + list((score[np.array(nbkg)-1] + score[np.array(nbkg)])/2) + [1] #find score in between two events which yield nbkg
  return boundaries

# def getBoundaries(bkg, nbkg, score_name="score", savepath=None):
#   #print(nbkg)
#   score = bkg[score_name].to_numpy()

#   def performFit(score, savepath):
#     # def fitFunction(x, a, b, c):
#     #   return a/(x-b-(a/c)) + c
#     # def fitFunction(x, a, b):
#     #   return a*x + b
#     def fitFunction(x, a):
#       c = 1 - a/2
#       return a*x + c
#     def fitFunctionCurve(x, a, b):
#       c = 1 + (1/a)*(np.exp(a*(1-b))-np.exp(-a*b))
#       return -np.exp(a*(x-b))+c
#     # def fitFunction(x, a, b):
#     #   def rhs(x, a, b):
#     #     c = 1 + (1/a)*(np.exp(a*(1-b))-np.exp(-a*b))
#     #     return -np.exp(a*(x-b))+c
#     #   def lhs(x, a, b):
#     #     b2 = b - 1
#     #     c = rhs(0.5, a, b) - np.exp(a*(-0.5-b2))
#     #     return np.exp(a*(-x-b)) + c

#     #   return np.where(x < 0.5, lhs(x, a, b), rhs(x, a, b))

#     score_half = score[score > 0.5]
#     score_half = (score_half - 0.5)*2
#     n, edges = np.histogram(score_half, range=(0, 1), bins=50, density=True)
#     n_unnormed, edges = np.histogram(score_half, range=(0, 1), bins=50, density=False)
    
#     #n, edges = np.histogram(score, range=(0, 1), bins=100, density=True)
#     #n_unnormed, edges = np.histogram(score, range=(0, 1), bins=100, density=False)
    
#     x = (edges[1:]+edges[:-1]) / 2
#     err = n / np.sqrt(n_unnormed)
    
#     #bounds = np.array([[0.05, 1, 0.5], [1, 5, 5]])
#     #popt, pcov = spo.curve_fit(fitFunction, fit_x, fit_n, sigma=fit_err, p0=[0.1, 1.5, 1], bounds=bounds)
#     #popt, pcov = spo.curve_fit(fitFunction, x, n, sigma=err, p0=[0, 1], bounds=[[-1, 0], [1, 10]])

#     popt, pcov = spo.curve_fit(fitFunction, x, n, sigma=err, p0=[0], bounds=[[-1], [1]])
#     chi2 = np.sum((n[40:]-fitFunction(x[40:], *popt))**2 / err[40:]**2) / len(x[40:])
#     print(savepath, chi2)
#     if chi2 < 0:
#       f = lambda x: 2*len(score_half)*fitFunction((x-0.5)*2, *popt)
#     else:
#       bounds = [[0, 0], [20, 2]]
#       popt, pcov = spo.curve_fit(fitFunctionCurve, x, n, sigma=err, p0=[10, 1.5], bounds=bounds)
#       f = lambda x: 2*len(score_half)*fitFunctionCurve((x-0.5)*2, *popt)


#     #bounds = [[0, 0], [20, 2]]
#     #popt, pcov = spo.curve_fit(fitFunction, x, n, sigma=err, p0=[10, 1.5], bounds=bounds)


#     if savepath is not None:
#       n, edges = np.histogram(score, range=(0, 1), bins=100, density=True)
#       n *= len(score)
#       n_unnormed, edges = np.histogram(score, range=(0, 1), bins=100, density=False)
#       x = (edges[1:]+edges[:-1]) / 2
#       err = n / np.sqrt(n_unnormed)

#       plt.clf()
#       plt.errorbar(x, n, err, fmt="ko")
#       plt.plot(x, f(x))
#       plt.ylim(bottom=0)
#       plt.savefig(savepath)
#       plt.clf()

#     return f

#   f = performFit(score, savepath)

#   N = 100000
#   x = np.linspace(1, 0, N)
#   cdf = np.cumsum(f(x)) / N
#   #print(cdf)
#   #cdf = spi.interp1d(x, cdf)
  
#   boundaries = [0] + [x[np.argmin(abs(cdf-n))] for n in nbkg] + [1]
#  # print(boundaries)

#   return boundaries

def remove_sr(df, sr):
  return df[~((df.Diphoton_mass>=sr[0]) & (df.Diphoton_mass<=sr[1]))]

def get_N_sidebands(nbkg):
  n = nbkg[::-1]
  return [n[0]] + [n[i+1]-n[i] for i in range(len(nbkg)-1)]

def resampleData(bkg, i=None):
  def fitFunction(x, a, b, c):
    return a/(x-b) + c
  
  def generateToys(f, a, b, n):
    poss_toys = np.random.rand(n*2)*(b-a) + a
    p = f(poss_toys)
    p /= sum(p)
    return np.random.choice(poss_toys, size=n, p=p, replace=False)
  
  bkgc = bkg[bkg.score > 0.5]
  bkgc.loc[:, "weight"] /= bkg.weight.sum()
  r = (0.5, 1.0)
  nbins = 50
  sumw, bins = np.histogram(bkgc.score, range=r, bins=nbins, weights=bkgc.weight)
  x = (bins[:-1]+bins[1:])/2
  popt, pcov = spo.curve_fit(fitFunction, x, sumw, p0=[1, 1.5, 1], bounds=[[0, 1+1e-3, 0], [100, 10, 10]])
  plt.scatter(x, sumw)
  plt.plot(x, fitFunction(x, *popt))
  plt.ylim(bottom=0)
  if i is None:
    plt.savefig("data_score_fit.png")
  else:
    plt.savefig("data_score_fit_%s.png"%i)
  plt.clf()

  toys = generateToys(lambda x: fitFunction(x, *popt), 0.5, 1.0, len(bkgc))
  toys = np.sort(toys)[::-1]
  bkgc.loc[:, "score"] = toys
  bkgc.loc[:, "weight"] = 1
  return bkgc

# def makeDataLikeBkg(bkg):
#   cdf = np.cumsum(bkg.weight.to_numpy())

#   print("start")
#   ids_to_keep = []
#   for i in range(int(cdf[-1])):
#     ids_to_keep.append(np.argmin(np.abs(cdf-i)))
#   print("end")

#   print(ids_to_keep[:10])
#   print(bkg.iloc[ids_to_keep].index)
#   pd.set_option('display.max_rows', 500)
#   print(bkg.iloc[:100])
  
#   all_ids = np.arange(0, len(bkg))
#   ids_to_drop = all_ids[~np.isin(all_ids, ids_to_keep)]

#   index_to_drop = bkg.iloc[ids_to_drop].index

#   bkg_copy = bkg.copy()

#   bkg.drop(index=index_to_drop, inplace=True)
#   bkg.loc[:,"weight"] = 1
#   print(bkg)

#   for i in range(30):
#     threshold = bkg.iloc[i].score
#     print(i, bkg_copy[bkg_copy.score > threshold].weight.sum())
#   return bkg

def makeDataLikeBkg(bkg, savepath=None):
  def performFit(score, weights, savepath):
    def fitFunction(x, a, b):
      c = 1 + (1/a)*(np.exp(a*(1-b))-np.exp(-a*b))
      return -np.exp(a*(x-b))+c

    json_path = savepath+".json"
    if os.path.exists(json_path):
      print("Loading json fit")
      with open(json_path, "r") as fi:
        fit = json.load(fi)
      
      f = lambda x: fit["norm"] * fitFunction((x-0.5)*2, *fit["popt"])
    else:
      s = score > 0.5
      sumw, edges = np.histogram((score[s]-0.5)*2, range=(0, 1), bins=20, weights=weights[s], density=True)
      sumw_unnormed, edges = np.histogram((score[s]-0.5)*2, edges, weights=weights[s], density=False)
      sumw2, edges = np.histogram((score[s]-0.5)*2, edges, weights=(weights[s]*(sumw[0]/sumw_unnormed[0]))**2, density=False)

      x = (edges[1:]+edges[:-1]) / 2
      err = np.sqrt(sumw2)

      err_per_sqrt_sumw = np.mean(err/np.sqrt(sumw))
      err = err_per_sqrt_sumw * np.sqrt(sumw)
      
      bounds = [[0, 0], [20, 2]]
      popt, pcov = spo.curve_fit(fitFunction, x, sumw, sigma=err, p0=[10, 1.5], bounds=bounds)
      print(popt)
      perr = np.sqrt(np.diag(pcov))

      fit = {"norm": 2*weights[s].sum(),
             "popt": list(popt)}
      with open(json_path, "w") as fi:
        fit = json.dump(fit, fi)
    
      f = lambda x: 2*weights[s].sum()*fitFunction((x-0.5)*2, *popt)

      if savepath is not None:
        sumw, edges = np.histogram(score, range=(0, 1), bins=40, weights=weights, density=True)
        sumw_unnormed, edges = np.histogram(score, edges, weights=weights, density=False)
        sumw2, edges = np.histogram(score, edges, weights=(weights*(sumw[0]/sumw_unnormed[0]))**2, density=False)

        x = (edges[1:]+edges[:-1]) / 2
        err = np.sqrt(sumw2)

        #err_per_sqrt_sumw = np.mean(err/np.sqrt(sumw))
        err_per_sqrt_sumw = np.sqrt(sumw2.sum()) / np.sqrt(sumw.sum())
        err = err_per_sqrt_sumw * np.sqrt(sumw)

        sumw *= weights.sum()
        err *= weights.sum()

        plt.clf()
        plt.errorbar(x, sumw, err, fmt="ko")
        xx = np.linspace(0, 1, 100)
        plt.plot(xx, f(xx))
        plt.ylim(bottom=0)
        plt.savefig(savepath)
        plt.clf()

    return f

  f = performFit(bkg.score, bkg.weight, savepath)

  N = 100000
  x = np.linspace(1, 0, N)
  cdf = np.cumsum(f(x)) / N
  cdf_inv = spi.interp1d(cdf, x)
  cdf = spi.interp1d(x, cdf)

  # if savepath is not None:
  #   x = np.linspace(0.8, 1.0, 100)
  #   plt.plot(x, cdf(x))
  #   plt.savefig(savepath+"_cdf")
  #   plt.clf()

  #   x = np.linspace(1, 100, 100)
  #   plt.plot(x, cdf_inv(x))
  #   plt.savefig(savepath+"_cdf_inv")
  #   plt.clf()

  Ns = np.arange(1, int(cdf(0)), 1)
  scores_at_Ns = cdf_inv(Ns)

  # keep background events which have score closest to points where cdf takes integer value
  bkg.reset_index(drop=True, inplace=True)

  idx_to_keep = np.searchsorted(bkg.score.to_numpy()[::-1], scores_at_Ns)

  # if the same event is found then skip to the next one
  while True:
    u, c = np.unique(idx_to_keep, return_counts=True)
    if (sum(c>1) == 0):
      break
    
    idx_to_increment = [np.where(idx_to_keep == i)[0][0] for i in u[c>1]]
    idx_to_keep[idx_to_increment] += 1
  assert max(idx_to_keep) < len(bkg) # this procedure may fail if we add beyond the length of dataframe

  all_idx = np.arange(0, len(bkg))
  idx_to_drop = all_idx[~np.isin(all_idx, idx_to_keep)]
  bkg.drop(idx_to_drop, inplace=True)

  # give events weight of 1
  bkg.loc[:, "weight"] = 1
  # place events in the middle of the scores that correspond to integers
  bkg.loc[:, "score"] = (np.concatenate([[1], scores_at_Ns[:-1]]) + scores_at_Ns) / 2

  return bkg

def optimiseBoundaries(args, bkg, sigs):
  pres = get_pres(args.sig_procs[0])
  srs = [get_sr(sig_proc) for sig_proc in args.sig_procs]

  df_package = lambda df, sig_proc: pd.DataFrame({"year":df.year, "mass":df.Diphoton_mass, "weight":df.weight, "score":df["%s_%s"%(args.score, sig_proc)]})

  bkgs_to_optim = [df_package(remove_sr(bkg, srs[i]), sig_proc) for i, sig_proc in enumerate(args.sig_procs)]
  sigs_to_optim = [df_package(sigs[sig_proc], sig_proc) for sig_proc in args.sig_procs]
  
  # sort score ready for getBoundaries function 
  for i, bkg_to_optim in enumerate(bkgs_to_optim):
    print(args.sig_procs[i])
    bkg_to_optim.sort_values("score", ascending=False, inplace=True)
    if args.mc_as_bkg:
      makeDataLikeBkg(bkg_to_optim, os.path.join(args.outdir, args.sig_procs[i]))
    
    #bkgs_to_optim[i] = resampleData(bkg_to_optim, args.sig_procs[i])

  max_n = min([len(each) for each in bkgs_to_optim]) #maximum number to define categories by  

  if args.set_nbkg is None:
    nbkg = [args.start_num]
    optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], nbkg, savepath=os.path.join(args.outdir, args.sig_procs[i]+".png")), i)[0] for i in range(len(args.sig_procs))])
    print(optimal_limits)
    to_add = args.start_num

    reached_end = False
    while (nbkg[0]+to_add < max_n):
      print()
      print(f">> {len(nbkg)} categories, with N_sidebands: {get_N_sidebands(nbkg)}", flush=True)
      
      found_improvement = False
      while (not found_improvement) and (nbkg[0]+to_add < max_n):
        print(f"> Considering adding {to_add} events")
        new_nbkg = [nbkg[0]+to_add] + nbkg
        new_optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], new_nbkg))[0] for i in range(len(args.sig_procs))])
        improve_frac = (optimal_limits-new_optimal_limits)/optimal_limits
        found_improvement = (improve_frac >= args.frac_improve).sum() >= args.n_proc_improve
        
        to_add *= 2
      
      if found_improvement:
        to_add //= 2 # if found acceptable improvement, we want to restart considering same to_add number (undo *= 2 from above)
        print(f"> Added {to_add} events because the following signal processes showed >= {args.frac_improve*100}% limit improvement:")
        print(" "+"\n ".join(np.array(args.sig_procs)[np.where(improve_frac >= args.frac_improve)[0]]))
        nbkg = new_nbkg
        print(nbkg)
        optimal_limits = new_optimal_limits
  
    print(">> Reached end of the events. Optimisation Finished.")
  else:
    nbkg = [sum(args.set_nbkg[:i+1]) for i in range(len(args.set_nbkg))][::-1]
    print(nbkg)
    optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], nbkg))[0] for i in range(len(args.sig_procs))])
  
  p = [getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], nbkg)) for i in range(len(args.sig_procs))]
  optimal_limits = {args.sig_procs[i]:p[i][0] for i in range(len(optimal_limits))}
  nsigs = {args.sig_procs[i]:p[i][2] for i in range(len(optimal_limits))}
  nbkgs = {args.sig_procs[i]:p[i][3] for i in range(len(optimal_limits))}
  
  boundaries = {args.sig_procs[i]: getBoundaries(bkgs_to_optim[i], nbkg) for i in range(len(args.sig_procs))}

  return nbkg, optimal_limits, nsigs, nbkgs, boundaries

def getOptimResults(args, bkg, optimal_boundaries, optimal_limits, nsigs, nbkgs, boundaries=None):
  optim_results = []
  scores = sorted(list(filter(lambda x: args.score in x, bkg.columns)))

  df_package = lambda df, sig_proc: pd.DataFrame({"year":df.year, "mass":df.Diphoton_mass, "weight":df.weight, "score":df["%s_%s"%(args.score, sig_proc)]})
  for i, score in enumerate(scores):
    print(score)

    sig_proc = score.split(args.score)[1][1:]
    sr = get_sr(sig_proc)

    bkg_to_optim = df_package(remove_sr(bkg, sr), sig_proc)
    bkg_to_optim.sort_values("score", ascending=False, inplace=True)
    if args.mc_as_bkg:
      makeDataLikeBkg(bkg_to_optim, os.path.join(args.outdir, sig_proc))

    results = {
      "sig_proc": sig_proc, 
      "score": score,
      "category_boundaries": getBoundaries(bkg_to_optim, optimal_boundaries)
    }
    if boundaries is not None:
      if sig_proc in boundaries.keys():
        assert (boundaries[sig_proc] == results["category_boundaries"])
      
    if sig_proc in optimal_limits.keys():
      results["optimal_limit"] = optimal_limits[sig_proc]
      results["nsigs"] = nsigs[sig_proc]
      results["nbkgs"] = nbkgs[sig_proc]

    optim_results.append(results)
  return optim_results

def main(args):
  if args.do_bootstrap != 0:
    for i in range(args.do_bootstrap):
      args_copy = copy.deepcopy(args)
      args_copy.do_bootstrap = 0
      args_copy.bootstrap_i = i+1
      args_copy.outdir = os.path.join(args.outdir, "bootstrap_%d"%(i+1)) 
      main(args_copy)

  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
    return True

  args.sig_procs = common.expandSigProcs(args.sig_procs)

  bkg, sigs, proc_dict = loadDataFrame(args)

  os.makedirs(args.outdir, exist_ok=True)
  nbkg, optimal_limits, nsigs, nbkgs, boundaries = optimiseBoundaries(args, bkg, sigs)
  optim_results = getOptimResults(args, bkg, nbkg, optimal_limits, nsigs, nbkgs, boundaries)
  
  print(">> Expected limits:")
  for sig_proc in sorted(optimal_limits.keys()):
    print((" %s: "%sig_proc).ljust(25), "%.3f"%optimal_limits[sig_proc])

  with open(os.path.join(args.outdir, "optim_results.json"), "w") as f:
    json.dump(optim_results, f, indent=4, sort_keys=True, cls=common.NumpyEncoder)

  with open(os.path.join(args.outdir, "N_in_sidebands.json"), "w") as f:
    N_in_sidebands = get_N_sidebands(nbkg)
    json.dump(N_in_sidebands, f)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str)
  parser.add_argument('--sig-procs', '-p', type=str, required=True, nargs="+")
  parser.add_argument('--score', type=str, default="score")

  parser.add_argument('--start-num', '-n', type=int, default=10)
  parser.add_argument('--n-proc-improve', type=int, default=1, help="At least this number of sig procs must improve to add new category")
  parser.add_argument('--frac-improve', type=float, default=0.01, help="Limits must improve by this fraction to accept new category")

  parser.add_argument('--do-bootstrap', type=int, default=0)
  parser.add_argument('--bootstrap-i', type=int, default=0)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)

  parser.add_argument('--set-nbkg', type=int, nargs="+", default=None)
  parser.add_argument('--mc-as-bkg', action="store_true")
  args = parser.parse_args()

  df = main(args)




