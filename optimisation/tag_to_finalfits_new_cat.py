import pandas as pd
import numpy as np
import json 
import argparse
import os
import uproot
import common

"""
Given one set of category boundaries, for every mass point (including interpolated)
I need to output a data root file and also a signal root file if the sig MC exists
"""

lumi_table = {
  2016: 35.9,
  2017: 41.5,
  2018: 59.8
}
lumi_table["combined"] = lumi_table[2016] + lumi_table[2017] + lumi_table[2018] 

def assignSignalRegions(df, optim_results, score_name):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1])
    df.loc[selection, "SR"] = i
  return df[df.SR!=-1]

def writeOutputTree(mass, weight, process, cat_name, year, undo_lumi_scaling=False, scale_signal=False):
  df = pd.DataFrame({"dZ": np.zeros(len(mass)), "CMS_hgg_mass": mass, "weight": weight})
  
  if undo_lumi_scaling:
    df.loc[:,"weight"] /= lumi_table[year]
  if scale_signal:
    df.loc[:,"weight"] /= 1000

  print(process, cat_name, year, df.weight.sum())

  path = os.path.join(args.outdir, "outputTrees", str(year))
  os.makedirs(path, exist_ok=True)
  with uproot.recreate(os.path.join(path, "%s_13TeV_%s_%s.root"%(process, cat_name, year))) as f:
    f["%s_13TeV_%s"%(process, cat_name)] = df

def main(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.optim_results) as f:
    optim_results = json.load(f)
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  for entry in optim_results:
    score_name = entry["score"]
    m = int(score_name.split("M")[1])

    tagged_df = assignSignalRegions(df, entry, score_name)
    data = tagged_df[tagged_df.process_id == proc_dict["Data"]]
    
    # if "XToHHggTauTau_M%d"%m in proc_dict.keys():
    #   sig = tagged_df[tagged_df.process_id == proc_dict["XToHHggTauTau_M%d"%m]]
    # else:
    #   sig = None
    # if args.injectSignal != "":
    #   sig_inject = tagged_df[tagged_df.process_id == proc_dict[args.injectSignal]]
    #   proc_name_inject = "gravitonm%d"%(common.get_MX_MY(args.injectSignal)[0])
    # else:
    #   sig_inject = None

    proc_name = "gravitonm%d"%m
    mgg = 125
    years = data.year.unique()

    for i, year in enumerate(years):
      for SR in tagged_df.SR.unique():
        if args.combineYears and (i==0):
          writeOutputTree(data[(data.SR==SR)].Diphoton_mass, data[(data.SR==SR)].weight, "Data", "%scat%d"%(proc_name, SR), "combined")
        elif not args.combineYears:
          writeOutputTree(data[(data.SR==SR)&(data.year==year)].Diphoton_mass, data[(data.SR==SR)&(data.year==year)].weight, "Data", "%scat%d"%(proc_name, SR), year)

        #if sig is not None:  writeOutputTree(sig[(sig.SR==SR)&(sig.year==year)].Diphoton_mass, sig[(sig.SR==SR)&(sig.year==year)].weight, "%s_%d_%d"%(proc_name, year, mgg), "%scat%d"%(proc_name, SR), year, undo_lumi_scaling=True, scale_signal=False)
        #if sig_inject is not None: writeOutputTree(sig_inject[(sig_inject.SR==SR)&(sig_inject.year==year)].Diphoton_mass, sig_inject[(sig_inject.SR==SR)&(sig_inject.year==year)].weight, "%s_%d_%d"%(proc_name, year, mgg), "%scat%d"%(proc_name, SR), year, undo_lumi_scaling=True, scale_signal=False)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--injectSignal', type=str, default="")
  parser.add_argument('--combineYears', action="store_true", help="Output data merged across years")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)