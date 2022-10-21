import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep

import scipy.interpolate as spi

from signalModelling.signal_fit import fitDCB
import signalModelling.signal_fit as signal_fit

import argparse
import os
import json
import common
import sys

mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

def fitSignalModel(df, mx, my, outdir):  
  fit_range = [my-10*(my/100.), my+10*(my/100.)]
  #fit_range = [112.5, 137.5]

  popt, perr = fitDCB(df[(df.MX==mx)&(df.MY==my)], fit_range=fit_range, savepath=os.path.join(outdir, "mx_%d_my_%d.png"%(mx,my)))
  
  popt[1] = popt[1] - my #change mean to a delta (dm)
  #popt[1] = popt[1] - 125

  return popt, perr

def tagSignals(df, threshold, threshold2, proc_dict, sig_proc=None):
  df.loc[:, "SR"] = -1

  if sig_proc is None:
    scores = list(filter(lambda x: "intermediate" in x, df.columns))
    for score in scores:
      proc = score.split("intermediate_transformed_score_")[1]
      if proc in proc_dict.keys():
        selection = (df[score] >= threshold) & (df[score] <= threshold2) & (df.process_id==proc_dict[proc])
        df.loc[selection, "SR"] = 0
  else:
    score = "intermediate_transformed_score_%s"%sig_proc
    selection = (df[score] >= threshold) & (df[score] <= threshold2)
    df.loc[selection, "SR"] = 0

  return df[df.SR!=-1]

def plot(m, p, perr, xlabel, ylabel, savename):
  s = np.argsort(m)
  m, p, perr = m[s], p[s], perr[s]

  plt.errorbar(m, p, perr, fmt='o')
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(savename)
  plt.clf()

def main(args):
  columns = common.getColumns(args.parquet_input)
  columns = list(filter(lambda x: (x[:5]!="score") and ("weight" not in x), columns))
  columns += ["weight"]
  #print(columns)
  df = pd.read_parquet(args.parquet_input, columns=columns)
  df = df[df.y==1]
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  common.add_MX_MY(df, proc_dict)
  
  df = tagSignals(df, args.threshold, args.threshold2, proc_dict, args.sig_proc)
  
  masses = []
  popts = []
  perrs = []

  for proc in df.process_id.unique()[:]:
    df_proc = df[df.process_id==proc]
    MX = df_proc.iloc[0].MX
    MY = df_proc.iloc[0].MY
    print(MX, MY)

    masses.append([MX, MY])
    popt, perr = fitSignalModel(df, MX, MY, args.outdir)

    norm = df_proc.weight.sum()
    norm /= sum([common.lumi_table[year] for year in common.lumi_table.keys()])
    norm_err = norm / np.sqrt(len(df_proc))
    popt = np.append(popt, norm)
    perr = np.append(perr, norm_err)

    popts.append(popt)
    perrs.append(perr)

  masses = np.array(masses)
  popts = np.array(popts)
  perrs = np.array(perrs)

  parameter_names = [r"$\Delta {m}_{\gamma\gamma}$", r"$\sigma$", r"$\beta_l$", r"$m_l$", r"$\beta_r$", r"$m_r$", "Signal Efficiency"]

  for MX in np.unique(masses[:,0]):
    for i in range(1,len(parameter_names)+1):
      plot(masses[masses[:,0]==MX,1], popts[masses[:,0]==MX,i], perrs[masses[:,0]==MX,i], r"$m_Y$", parameter_names[i-1], os.path.join(args.outdir, "mx_%d_p%d"%(MX, i)))

  for MY in np.unique(masses[:,1]):
    for i in range(1,len(parameter_names)+1):
      plot(masses[masses[:,1]==MY,0], popts[masses[:,1]==MY,i], perrs[masses[:,1]==MY,i], r"$m_X$", parameter_names[i-1], os.path.join(args.outdir, "my_%d_p%d"%(MY, i)))

  

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--threshold', '-t', type=float, default=0.99)
  parser.add_argument('--threshold2', type=float, default=1.0)

  parser.add_argument('--sig-proc', '-p', type=str, default=None, help="Use this process's score as opposed to a tailored score per process.")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)