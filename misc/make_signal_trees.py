import pandas as pd
import numpy as np
import json 
import argparse
import os
import uproot
import common
import tabulate
import sys
import common

def assignSignalRegions(df, optim_results, score_name):
  df["SR"] = -1
  
  boundaries = optim_results["category_boundaries"][::-1] #so that cat0 is most pure
  for i in range(len(boundaries)-1):
    selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1])
    df.loc[selection, "SR"] = i
  return df[df.SR!=-1]

# def writeOutputTree(mass, weight, process, cat_name, year, undo_lumi_scaling=False, scale_signal=False):
#   df = pd.DataFrame({"dZ": np.zeros(len(mass)), "CMS_hgg_mass": mass, "weight": weight})
  
#   if undo_lumi_scaling:
#     df.loc[:,"weight"] /= common.lumi_table[year]
#   if scale_signal:
#     df.loc[:,"weight"] /= 1000

#   path = os.path.join(args.outdir, "outputTrees", str(year))
#   os.makedirs(path, exist_ok=True)
#   with uproot.recreate(os.path.join(path, "%s_13TeV_%s_%s.root"%(process, cat_name, year))) as f:
#     f["%s_13TeV_%s"%(process, cat_name)] = df

def createDataframeTree(mass, weight, process, cat_name, year, undo_lumi_scaling=False, scale_signal=False):
  df = pd.DataFrame({"dZ": np.zeros(len(mass)), "CMS_hgg_mass": mass, "weight": weight})
  
  if undo_lumi_scaling:
    df.loc[:,"weight"] /= common.lumi_table[year]
  if scale_signal:
    df.loc[:,"weight"] /= 1000

  return df

def write(dfs, process, cat_name, year, names):
  path = os.path.join(args.outdir, "outputTrees", str(year))
  os.makedirs(path, exist_ok=True)
  with uproot.recreate(os.path.join(path, "%s_125_13TeV_%s_%s.root"%(process, cat_name, year))) as f:
    for i, df in enumerate(dfs):
      name = names[i]
      if name == "nominal":
        f["%s_125_13TeV_%s"%(process, cat_name)] = df
      else:
        f["%s_125_13TeV_%s_%s"%(process, cat_name, name)] = df

def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=4)
    return True

  with open(args.optim_results) as f:
    optim_results = json.load(f)
  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  columns = common.getColumns(os.path.join(args.parquet_dir, "merged_nominal.parquet"))
  columns = list(filter(lambda x: ("score" not in x) and ("weight" not in x), columns))
  columns += ["intermediate_transformed_score_NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_600", "weight"]
  print(columns)

  select_sig = lambda df: df[df.process_id==proc_dict["NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_600"]]

  dfs = {}

  dfs["nominal"] = select_sig(pd.read_parquet(os.path.join(args.parquet_dir, "merged_nominal.parquet"), columns=columns))
  
  systematics = ["material", "fnuf", "scale", "smear"]
  syst_dfs = {}
  for systematic in systematics:
    print(systematic)
    dfs["%sUp01sigma"%systematic] = select_sig(pd.read_parquet(os.path.join(args.parquet_dir, "merged_%s_up.parquet"%systematic), columns=columns))
    dfs["%sDown01sigma"%systematic] = select_sig(pd.read_parquet(os.path.join(args.parquet_dir, "merged_%s_down.parquet"%systematic), columns=columns))

  for entry in optim_results:
    MX, MY = common.get_MX_MY(entry["sig_proc"])
    if not (MX==1000 and MY==600): continue

    print("Tagging")
    for name in dfs:
      dfs[name] = assignSignalRegions(dfs[name], entry, entry["score"])

    proc_name = "ggttresmx%dmy%d"%(MX, MY)
    years = dfs["nominal"].year.unique()    

    for i, year in enumerate(years):
      SRs = np.sort(dfs[name].SR.unique())
      if args.dropLastCat: SRs = SRs[:-1]
      
      for SR in SRs:
        names_list = [name for name in dfs]
        dfs_list = [dfs[name] for name in names_list]

        dfTrees = [createDataframeTree(df[(df.SR==SR)&(df.year==year)].Diphoton_mass, df[(df.SR==SR)&(df.year==year)].weight, proc_name, "%scat%d"%(proc_name, SR), year, undo_lumi_scaling=True) for df in dfs_list]
        write(dfTrees, proc_name, "%scat%d"%(proc_name, SR), year, names_list)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-dir', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--combineYears', action="store_true", help="Output data merged across years")
  parser.add_argument('--dropLastCat', action="store_true")
  parser.add_argument('--justYields', action="store_true")
  parser.add_argument('--batch', action="store_true")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  df = main(args)