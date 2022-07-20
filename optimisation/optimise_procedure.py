import pandas as pd
import numpy as np
import argparse
import os
import json

from optimisation.limit import getBoundariesPerformance

def loadDataFrame(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  data = df[df.process_id == proc_dict["Data"]]
  sigs = {sig_proc: df[df.process_id == proc_dict[sig_proc]] for sig_proc in args.sig_procs}

  return data, sigs, proc_dict

def getBoundaries(bkg, nbkg, score_name="score"):
  return [0] + list(bkg[score_name].to_numpy()[nbkg] - 1e-8) + [1]

def hasEnoughSig(sigs_to_optim, bkgs_to_optim, nbkg):
  select = lambda df, i: df[(df.score > boundaries[i]) & (df.score <= boundaries[i+1])]
  nCats = len(nbkg) + 1

  for j in range(nCats):
    enoughSignal = False
    maxSignal = 0
    for i in range(len(sigs_to_optim)):
      boundaries = getBoundaries(bkgs_to_optim[i], nbkg)
      df = sigs_to_optim[i][(sigs_to_optim[i].score > boundaries[j]) & (sigs_to_optim[i].score <= boundaries[j+1])]

      years, counts = np.unique(df.year, return_counts=True)
      #print(counts)
      if np.all(counts>=200):
        enoughSignal = True
        if min(counts) > maxSignal: maxSignal = min(counts)
    if not enoughSignal:
      return False  
    print(j, maxSignal)
  return True

def optimiseBoundaries(args, data, sigs, proc_dict):
  bkgs_to_optim = [pd.DataFrame({"year":data.year, "mass":data.Diphoton_mass, "weight":data.weight, "score":data["%s_%s"%(args.score, sig_proc)]}) for sig_proc in args.sig_procs]
  sigs_to_optim = [pd.DataFrame({"year":sigs[sig_proc].year, "mass":sigs[sig_proc].Diphoton_mass, "weight":sigs[sig_proc].weight, "score":sigs[sig_proc]["%s_%s"%(args.score, sig_proc)]}) for sig_proc in args.sig_procs]

  for bkg_to_optim in bkgs_to_optim:
    bkg_to_optim.sort_values("score", ascending=False, inplace=True)    

  nbkg = [19, 9]
  optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], args.pres, args.sr, getBoundaries(bkgs_to_optim[i], nbkg))[0] for i in range(len(args.sig_procs))])
  to_add = 10

  reached_end = False
  while not reached_end:
    print(nbkg)
    enough_improvement = False
    while not enough_improvement:

      if nbkg[0]+to_add > len(data):
          reached_end = True
          break

      new_nbkg = [nbkg[0]+to_add] + nbkg
      new_optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], args.pres, args.sr, getBoundaries(bkgs_to_optim[i], new_nbkg))[0] for i in range(len(args.sig_procs))])

      if (max((optimal_limits-new_optimal_limits)/optimal_limits) < 0.02):
        to_add *= 2
        print(to_add)
      else:
        nbkg = new_nbkg
        optimal_limits = new_optimal_limits
        enough_improvement = True

  optimal_limits = {args.sig_procs[i]:optimal_limits[i] for i in range(len(optimal_limits))}

  return nbkg, optimal_limits

def getOptimResults(args, data, optimal_boundaries, optimal_limits):
  optim_results = []
  scores = sorted(list(filter(lambda x: args.score in x, data.columns)))
  for i, score in enumerate(scores):
    data.sort_values(score, ascending=False, inplace=True)

    sig_proc = score.split(args.score)[1][1:]
    results = {
      "sig_proc": sig_proc, 
      "score": score,
      "category_boundaries": getBoundaries(data, optimal_boundaries, score)
    }
    if sig_proc in optimal_limits.keys():
      results["optimal_limit"] = optimal_limits[sig_proc]

    optim_results.append(results)
  return optim_results

def main(args):
  data, sigs, proc_dict = loadDataFrame(args)

  bm = data.Diphoton_mass
  sidebands = ((bm > args.pres[0]) & (bm < args.sr[0])) | ((bm > args.sr[1]) & (bm < args.pres[1]))
  data = data[sidebands]

  optimal_boundaries, optimal_limits = optimiseBoundaries(args, data, sigs, proc_dict)
  print(optimal_boundaries)
  print(optimal_limits)
  optim_results = getOptimResults(args, data, optimal_boundaries, optimal_limits)
  
  with open(args.out, "w") as f:
    json.dump(optim_results, f, indent=4)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--out', '-o', type=str)
  parser.add_argument('--sig-procs', '-p', type=str, required=True, nargs="+")
  parser.add_argument('--pres', type=float, default=(100,180), nargs=2)
  parser.add_argument('--sr', type=float, default=(115,135), nargs=2)
  parser.add_argument('--score', type=str, default="score")
  args = parser.parse_args()

  df = main(args)




