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

  df = pd.read_parquet(args.parquet_input, columns=columns_load)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  data = df[df.process_id == proc_dict["Data"]]
  sigs = {sig_proc: df[df.process_id == proc_dict[sig_proc]] for sig_proc in args.sig_procs}

  bkg = data # use data as background
  
  if args.bootstrap_i != 0:
    rng = np.random.default_rng(seed=args.bootstrap_i)
    bkg = bkg.loc[rng.choice(bkg.index, size=len(bkg), replace=True)]

  return bkg, sigs, proc_dict

def getBoundaries(bkg, nbkg, score_name="score"):
  """
  Return boundaries which correspond to events found at indexes given by nbkg.
  The intention is that the bkg dataframe is already sorted by score such that the boundaries
  correspond to selecting the nbkg[0]-highest scoring events, followed by the next nbkg[1]...
  """
  score = bkg[score_name].to_numpy()
  print(score[:20])
  boundaries = [0] + list((score[np.array(nbkg)-1] + score[np.array(nbkg)])/2) + [1] #find score in between two events which yield nbkg
  return boundaries

def remove_sr(df, sr):
  return df[~((df.Diphoton_mass>=sr[0]) & (df.Diphoton_mass<=sr[1]))]

def get_N_sidebands(nbkg):
  n = nbkg[::-1]
  return [n[0]] + [n[i+1]-n[i] for i in range(len(nbkg)-1)]

def optimiseBoundaries(args, bkg, sigs):
  pres = get_pres(args.sig_procs[0])
  srs = [get_sr(sig_proc) for sig_proc in args.sig_procs]

  df_package = lambda df, sig_proc: pd.DataFrame({"year":df.year, "mass":df.Diphoton_mass, "weight":df.weight, "score":df["%s_%s"%(args.score, sig_proc)]})

  bkgs_to_optim = [df_package(remove_sr(bkg, srs[i]), sig_proc) for i, sig_proc in enumerate(args.sig_procs)]
  sigs_to_optim = [df_package(sigs[sig_proc], sig_proc) for sig_proc in args.sig_procs]
  
  max_n = min([len(each) for each in bkgs_to_optim]) #maximum number to define categories by

  # sort score ready for getBoundaries functio 
  for bkg_to_optim in bkgs_to_optim:
    bkg_to_optim.sort_values("score", ascending=False, inplace=True)    

  if args.set_nbkg is None:
    nbkg = [args.start_num]
    optimal_limits = np.array([getBoundariesPerformance(bkgs_to_optim[i], sigs_to_optim[i], pres, srs[i], getBoundaries(bkgs_to_optim[i], nbkg), i)[0] for i in range(len(args.sig_procs))])
    print(optimal_limits)
    to_add = args.start_num

    reached_end = False
    while (nbkg[0]+to_add < max_n):
      print()
      print(f">> {len(nbkg)} categories, with N_sidebands: {get_N_sidebands(nbkg)}")
      
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
    common.submitToBatch([sys.argv[0]] + common.parserToList(args))
    return True

  args.sig_procs = common.expandSigProcs(args.sig_procs)

  bkg, sigs, proc_dict = loadDataFrame(args)

  bm = bkg.Diphoton_mass
  pres = get_pres(args.sig_procs[0]) #is same for all signal processes in one analysis
  bkg = bkg[((bm >= pres[0]) & (bm <= pres[1]))]

  nbkg, optimal_limits, nsigs, nbkgs = optimiseBoundaries(args, bkg, sigs)
  optim_results = getOptimResults(args, bkg, nbkg, optimal_limits, nsigs, nbkgs)
  
  print(">> Expected limits:")
  for sig_proc in sorted(optimal_limits.keys()):
    print((" %s: "%sig_proc).ljust(25), "%.3f"%optimal_limits[sig_proc])

  os.makedirs(args.outdir, exist_ok=True)
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

  parser.add_argument('--set-nbkg', type=int, nargs="+", default=None)
  args = parser.parse_args()

  df = main(args)




