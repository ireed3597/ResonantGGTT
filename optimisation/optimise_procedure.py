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

  #df = df[df.year==2018]

  data = df[df.process_id == proc_dict["Data"]]
  sigs = {sig_proc: df[df.process_id == proc_dict[sig_proc]] for sig_proc in args.sig_procs}

  return data, sigs, proc_dict

def getBoundaries(bkg, nbkg):
  return [0] + list(bkg.score.to_numpy()[nbkg] - 1e-8) + [1]

def main(args):
  data, sigs, proc_dict = loadDataFrame(args)

  bm = data.Diphoton_mass
  sidebands = ((bm > args.pres[0]) & (bm < args.sr[0])) | ((bm > args.sr[1]) & (bm < args.pres[1]))
  data = data[sidebands]

  bkgs_to_optim = [pd.DataFrame({"mass":data.Diphoton_mass, "weight":data.weight, "score":data["%s_%s"%(args.score, sig_proc)]}) for sig_proc in args.sig_procs]
  sigs_to_optim = [pd.DataFrame({"mass":sigs[sig_proc].Diphoton_mass, "weight":sigs[sig_proc].weight, "score":sigs[sig_proc]["%s_%s"%(args.score, sig_proc)]}) for sig_proc in args.sig_procs]

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

      if max((optimal_limits-new_optimal_limits)/optimal_limits) < 0.01:
        to_add *= 2
      else:
        nbkg = new_nbkg
        optimal_limits = new_optimal_limits
        enough_improvement = True

  print(list(optimal_limits))

  

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str)
  parser.add_argument('--sig-procs', '-p', type=str, required=True, nargs="+")
  parser.add_argument('--pres', type=float, default=(100,150), nargs=2)
  parser.add_argument('--sr', type=float, default=(120,130), nargs=2)
  parser.add_argument('--score', type=str, default="score")
  args = parser.parse_args()

  if args.outdir == None:
    args.outdir = os.path.join("optimisation_output", args.sig_proc)

  os.makedirs(args.outdir, exist_ok=True)

  df = main(args)




