import pandas as pd
import numpy as np
import json 
import argparse
import os
import uproot
import common

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

def assignSignalRegions(df, optim_results, score_name):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[score_name] <= boundaries[i]) & (df[score_name] >= boundaries[i+1])
    df.loc[selection, "SR"] = i
  return df

def getXLimits(masses, sumws, cat_num):
  lows = []
  highs = []
  for i in range(len(sumws[:,0,0])):
    select = (sumws[i,:,cat_num] / max(sumws[i,:,cat_num])) > 0.5
    lows.append(np.array(masses)[select].min())
    highs.append(np.array(masses)[select].max())

  return min(lows), max(highs)

def main(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.optim_results) as f:
    optim_results = json.load(f)
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  score_prefix = "intermediate_transformed_score_XToHHggTauTau_M"
  masses = sorted([int(column[len(score_prefix):]) for column in df.columns if score_prefix in column])

  args.sig_procs = args.sig_procs[1:] + [args.sig_procs[0]]
  
  for k in range(0, len(args.sig_procs), 4):
    sumws = []
    for sig_proc in args.sig_procs[k:k+4]:
      print(sig_proc)
      sumw_sig_proc = []
      for m in masses:
        score_name = score_prefix + str(m)
        tagged_df = assignSignalRegions(df, optim_results, score_name)

        sumw_m = []    
        for SR in sorted(tagged_df.SR.unique()):
          sumw_m.append(tagged_df.loc[(tagged_df.SR==SR)&(tagged_df.process_id==proc_dict[sig_proc]), "weight"].sum())
        sumw_sig_proc.append(sumw_m)
      sumws.append(sumw_sig_proc)

    sumws = np.array(sumws)

    nCats = len(sumws[0,0])
    for i in range(nCats):
      for j, sig_proc in enumerate(args.sig_procs[k:k+4]):
        plt.plot(masses, sumws[j,:,i]/max(sumws[j,:,i]), label=sig_proc)
      plt.legend()
      plt.ylabel("Signal efficiency (normed to maximum)")
      plt.xlabel(r"$m_X$")
      plt.xlim(*getXLimits(masses, sumws, i))
      plt.savefig(os.path.join(args.outdir, "sumw_fadeaway_%d_cat%d.png"%(k,i)))
      plt.clf()

  return df, sumws, masses

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+")

  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  df, sumws, masses = main(args)