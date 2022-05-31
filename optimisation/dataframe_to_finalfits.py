import uproot
import pandas as pd
import numpy as np
import os
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lumi_table = {
  2016: 35.9,
  2017: 41.5,
  2018: 59.8
}

def loadDataFrame(args):
  df = pd.read_parquet(args.parquet_input)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  df = df[(df.Diphoton_mass>100) & (df.Diphoton_mass<180)]

  data = df[df.process_id == proc_dict["Data"]]
  sig = df[df.process_id == proc_dict[args.sig_proc]]
  bkg = df[(df.y==0) & (df.process_id != proc_dict["Data"])]

  return data, sig, bkg, proc_dict

def writeOutputTree(mass, weight, process, cat_name, year, undo_lumi_scaling=False, scale_signal=False):
  df = pd.DataFrame({"dZ": np.zeros(len(mass)), "CMS_hgg_mass": mass, "weight": weight})
  
  if undo_lumi_scaling:
    df.loc[:,"weight"] /= lumi_table[year]
  if scale_signal:
    df.loc[:,"weight"] /= 1000

  print(cat_name, df.weight.sum())

  path = os.path.join(args.outdir, "outputTrees", str(year))
  os.makedirs(path, exist_ok=True)
  with uproot.recreate(os.path.join(path, "%s_13TeV_%s_%s.root"%(process, cat_name, year))) as f:
    f["%s_13TeV_%s"%(process, cat_name)] = df

def main(args):
  data, sig, bkg, proc_dict = loadDataFrame(args)

  sig_s = sig[sig.process_id == proc_dict[args.sig_proc]]
  bkg = bkg[bkg.process_id == proc_dict["VH_M125"]]

  print("all", sig_s.weight.sum()/(59*1000))

  """
  if "radion" in args.sig_proc:
    mgg = 125
    mx = int(args.sig_proc.split("M")[1].split("_")[0])
    proc_name = "radionm%d"%mx
  elif "Y_gg" in args.sig_proc:
    mgg = int(args.sig_proc.split("_")[-1])
    mx = int(args.sig_proc.split("_")[-3])
    my = mgg
    proc_name = "NmssmYggMx%dMy%d"%(mx, my)
  """
  mgg = 125.0
  mx = 500
  proc_name="radionm500"

  years = data.year.unique()
  
  for year in years:
    print(year)
    for SR in data.SR.unique():
      print(SR)
      writeOutputTree(sig_s[(sig_s.SR==SR)&(sig.year==year)].Diphoton_mass, sig_s[(sig_s.SR==SR)&(sig.year==year)].weight, "%s_%d"%(proc_name, mgg), "%scat%d"%(proc_name, SR), year, undo_lumi_scaling=True, scale_signal=False)
      #writeOutputTree(bkg[bkg.SR==SR].Diphoton_mass, bkg[bkg.SR==SR].weight, "VH_125", "radionm%dcat%d"%(mass, SR), 2018, scale_signal=False)
      writeOutputTree(data[(data.SR==SR)&(data.year==year)].Diphoton_mass, data[(data.SR==SR)&(data.year==year)].weight, "Data", "%scat%d"%(proc_name, SR), year)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str)
  parser.add_argument('--sig-proc', '-p', type=str, required=True)
  args = parser.parse_args()

  if args.outdir == None:
    args.outdir = os.path.join("tagging_output", args.sig_proc)

  main(args)