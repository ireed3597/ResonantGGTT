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

def loadDataFrame(args):
  columns = common.getColumns(args.parquet_input)
  columns_load = ["Diphoton_mass", "weight", "year", "process_id"]
  columns_load += list(filter(lambda x: args.score in x, columns))
  #print(columns_load)

  df = pd.read_parquet(args.parquet_input, columns=columns_load)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  data = df[df.process_id == proc_dict["Data"]]
  sigs = {sig_proc: df[df.process_id == proc_dict[sig_proc]] for sig_proc in args.sig_procs}

  bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"] if proc in proc_dict.keys()]
  bkg = df[df.process_id.isin(bkg_ids)]

  if args.bkg_source == "data":
    bkg = data
  elif args.bkg_source == "MC":
    pass
  else:
    raise AttributeError("Background source must be 'data' or 'MC'")
  
  if args.bootstrap_i != 0:
    rng = np.random.default_rng(seed=args.bootstrap_i)
    bkg = bkg.loc[rng.choice(bkg.index, size=len(bkg), replace=True)]

  return bkg, sigs, proc_dict

def getBoundaries(bkg, nbkg, score_name="score", to_print=None):
  #print(to_print)
  score = bkg[score_name].to_numpy()
  #print(list(score[:15]))

  boundaries = [0] + list((score[np.array(nbkg)-1] + score[np.array(nbkg)])/2) + [1] #find score in between two events which yield nbkg
  #boundaries = [0] + list(score[np.array(nbkg)] - 1e-10) + [1] #find score in between two events which yield nbkg
  other_boundaries = boundaries
  
  # cdf = np.cumsum(bkg.weight.to_numpy())
  # idxs = np.arange(0, len(cdf))
  # other_boundaries = [1]
  
  # closest_cdf = 0
  # for i, n in enumerate(nbkg[::-1]):
  #   n_search = max([n, closest_cdf+10]) #have at least a spacing of 10 between events
  #   closest_cdf = cdf[min(idxs[(cdf-n_search)>=0])]
  #   other_boundaries.append((score[np.where(cdf==closest_cdf)[0][0]] + score[np.where(cdf==closest_cdf)[0][0]+1])/2)
  # other_boundaries.append(0)
  # other_boundaries = other_boundaries[::-1]

  #print(boundaries)
  #print(other_boundaries)

  #print( [sum((score > boundaries[i]) & (score <= boundaries[i+1])) for i in range(len(boundaries)-1)] )
  #print( [sum((score > other_boundaries[i]) & (score <= other_boundaries[i+1])) for i in range(len(other_boundaries)-1) ])

  return other_boundaries

remove_sr = lambda df, sr: df[~((df.Diphoton_mass>=sr[0]) & (df.Diphoton_mass<=sr[1]))]

def optimiseBoundaries(args, bkg, sigs):
  pres = get_pres(args.sig_procs[0])
  srs = [get_sr(sig_proc) for sig_proc in args.sig_procs] #signal region auto generated as MY +- 10 GeV if selected in args
  print(srs)

  df_package = lambda df, sig_proc: pd.DataFrame({"year":df.year, "mass":df.Diphoton_mass, "weight":df.weight, "score":df["%s_%s"%(args.score, sig_proc)]})

  bkgs_to_optim = [df_package(remove_sr(bkg, srs[i]), sig_proc) for i, sig_proc in enumerate(args.sig_procs)]
  sigs_to_optim = [df_package(sigs[sig_proc], sig_proc) for sig_proc in args.sig_procs]
  
  max_n = min([len(each) for each in bkgs_to_optim]) #maximum number to define categories by

  for bkg_to_optim in bkgs_to_optim:
    bkg_to_optim.sort_values("score", ascending=False, inplace=True)    

  nbkg = [args.start_num]
  optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], nbkg, to_print=args.sig_procs[i]))[0] for i in range(len(args.sig_procs))])
  to_add = args.start_num

  reached_end = False
  while not reached_end:
    print(nbkg)
    enough_improvement = False
    while not enough_improvement:

      if (nbkg[0]+to_add > max_n):
        reached_end = True
        break

      new_nbkg = [nbkg[0]+to_add] + nbkg
      new_optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], new_nbkg, to_print=args.sig_procs[i]))[0] for i in range(len(args.sig_procs))])

      improve_frac = (optimal_limits-new_optimal_limits)/optimal_limits
      if (improve_frac >= args.frac_improve).sum() < args.n_proc_improve:
        to_add *= 2
        print(to_add)
      else:
        print(np.array(args.sig_procs)[np.where(improve_frac >= args.frac_improve)[0]])
        nbkg = new_nbkg
        optimal_limits = new_optimal_limits
        enough_improvement = True

  #p contains limits, ams, nsigs, nbkgs
  p = [getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], nbkg)) for i in range(len(args.sig_procs))]
  optimal_limits = {args.sig_procs[i]:p[i][0] for i in range(len(optimal_limits))}
  nsigs = {args.sig_procs[i]:p[i][2] for i in range(len(optimal_limits))}
  nbkgs = {args.sig_procs[i]:p[i][3] for i in range(len(optimal_limits))}
  # for i, sig_proc in enumerate(args.sig_procs):
  #   print(sig_proc)
  #   print(getBoundaries(bkgs_to_optim[i], nbkg))

  return nbkg, optimal_limits, nsigs, nbkgs

def getOptimResults(args, bkg, optimal_boundaries, optimal_limits, nsigs, nbkgs):
  optim_results = []
  scores = sorted(list(filter(lambda x: args.score in x, bkg.columns)))
  for i, score in enumerate(scores):
    bkg.sort_values(score, ascending=False, inplace=True)

    sig_proc = score.split(args.score)[1][1:]
    sr = get_sr(sig_proc)

    results = {
      "sig_proc": sig_proc, 
      "score": score,
      "category_boundaries": getBoundaries(remove_sr(bkg, sr), optimal_boundaries, score)
    }
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
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=4)
    return True

  bkg, sigs, proc_dict = loadDataFrame(args)

  bm = bkg.Diphoton_mass
  pres = get_pres(args.sig_procs[0]) #is same for all signal processes in one analysis
  bkg = bkg[((bm >= pres[0]) & (bm <= pres[1]))]

  optimal_boundaries, optimal_limits, nsigs, nbkgs = optimiseBoundaries(args, bkg, sigs)
  print(optimal_boundaries)
  print(optimal_limits)
  optim_results = getOptimResults(args, bkg, optimal_boundaries, optimal_limits, nsigs, nbkgs)
  
  os.makedirs(args.outdir, exist_ok=True)
  with open(os.path.join(args.outdir, "optim_results.json"), "w") as f:
    json.dump(optim_results, f, indent=4, sort_keys=True, cls=common.NumpyEncoder)

  with open(os.path.join(args.outdir, "N_in_sidebands.json"), "w") as f:
    N_in_sidebands = [optimal_boundaries[-1]] + [optimal_boundaries[::-1][i]-optimal_boundaries[::-1][i-1] for i in range(1, len(optimal_boundaries))]
    print(N_in_sidebands)
    json.dump(N_in_sidebands, f)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str)
  parser.add_argument('--sig-procs', '-p', type=str, required=True, nargs="+")
  parser.add_argument('--score', type=str, default="score")

  parser.add_argument('--bkg-source', '-b', type=str, default="data")

  #parser.add_argument('--pres', type=float, default=(100,180), nargs=2)
  #parser.add_argument('--sr', type=float, default=(115,135), nargs=2)
  #parser.add_argument('--sr-MYpm10', action="store_true")

  parser.add_argument('--start-num', '-n', type=int, default=10)
  parser.add_argument('--n-proc-improve', type=int, default=1, help="At least this number of sig procs must improve to add new category")
  parser.add_argument('--frac-improve', type=float, default=0.01, help="Limits must improve by this fraction to accept new category")

  parser.add_argument('--do-bootstrap', type=int, default=0)
  parser.add_argument('--bootstrap-i', type=int, default=0)
  parser.add_argument('--batch', action="store_true")
  args = parser.parse_args()

  df = main(args)




