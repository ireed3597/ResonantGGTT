import pandas as pd
import numpy as np
import argparse
import os
import json

from optimisation.limit import optimiseBoundary
from optimisation.limit import transformScores

from optimisation.limit import getBoundariesPerformance

def loadDataFrame(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  data = df[df.process_id == proc_dict["Data"]]
  sig = df[df.process_id == proc_dict[args.sig_proc]]
  bkg = df[(df.y==0) & (df.process_id != proc_dict["Data"])]

  return data, sig, bkg, proc_dict

def main(args):
  data, sig, bkg, proc_dict = loadDataFrame(args)

  bkg_to_optim = pd.DataFrame({"mass":data.Diphoton_mass, "weight":data.weight, "score":data[args.score].astype(float)})
  sig_to_optim = pd.DataFrame({"mass":sig.Diphoton_mass, "weight":sig.weight, "score":sig[args.score].astype(float)})

  optimal_limit, optimal_boundaries, boundaries, limits, ams = optimiseBoundary(bkg_to_optim, sig_to_optim, pres=args.pres, sr=args.sr, low=args.low, high=args.high, step=args.step, nbounds=args.nbounds, include_lower=args.include_lower)

  optimal_limit = round(optimal_limit, 4)
  for i in range(len(optimal_boundaries)):
    optimal_boundaries[i] = round(optimal_boundaries[i], 10)

  print(optimal_limit, ams[np.argmin(limits)], list(optimal_boundaries))

  #compare = [0.0, 0.9972776671, 0.9991124856, 1.0]
  #print(getBoundariesPerformance(bkg_to_optim, sig_to_optim, args.pres, args.sr, compare))

  select = lambda df, pair: (df.score > pair[0]) & (df.score <= pair[1])
  bm = bkg_to_optim.mass
  sidebands = ((bm > args.pres[0]) & (bm < args.sr[0])) | ((bm > args.sr[1]) & (bm < args.pres[1]))
  sr = (bm > args.sr[0]) & (bm < args.sr[1])
  for i in range(len(optimal_boundaries)-1):
    nbkg_in_sidebands = sum(sidebands & select(bkg_to_optim, [optimal_boundaries[i], optimal_boundaries[i+1]]))
    nsig = sig_to_optim[select(sig_to_optim, [optimal_boundaries[i], optimal_boundaries[i+1]])].weight.sum()
    print(optimal_boundaries[i], optimal_boundaries[i+1], nbkg_in_sidebands, nsig)

  results = {}
  results["sig_proc"] = args.sig_proc
  results["score"] = args.score
  results["optimal_limit"] = optimal_limit
  results["category_boundaries"] = list(optimal_boundaries)
  with open(os.path.join(args.outdir, "optimisation_results.json"), "w") as f:
    json.dump(results, f, indent=4)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str)
  parser.add_argument('--sig-proc', '-p', type=str, required=True)
  parser.add_argument('--pres', type=float, default=(100,180), nargs=2)
  parser.add_argument('--sr', type=float, default=(115,135), nargs=2)
  parser.add_argument('--score', type=str, default="score")

  parser.add_argument('--low', type=float, default=0.1)
  parser.add_argument('--high', type=float, default=0.99)
  parser.add_argument('--step', type=float, default=0.01)
  parser.add_argument('--nbounds', type=int, default=1)
  parser.add_argument('--include-lower', action="store_true")
  args = parser.parse_args()

  if args.outdir == None:
    args.outdir = os.path.join("optimisation_output", args.sig_proc)
  
  args.score = "%s_%s"%(args.score, args.sig_proc)

  os.makedirs(args.outdir, exist_ok=True)

  #import cProfile
  #cProfile.run('df=main(args)', 'restats')

  df = main(args)




