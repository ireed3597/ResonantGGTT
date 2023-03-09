import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np
import sys
import os
import json
import common
import argparse
import os
import pandas as pd

from signalModelling.interpolate import tagSignals
from signalModelling.signal_fit import histogram, plotFitComparison

def plotChecks(df, proc_dict, optim_results, args):
  with open(os.path.join(args.outdir, "model.json"), "r") as f:
    nominal_model = json.load(f)

  check_sig_procs = os.listdir(os.path.join(args.outdir, "Interpolation_Check"))
  for sig_proc in check_sig_procs:
    mx, my = common.get_MX_MY(sig_proc)
    print(mx)

    for each in optim_results:
      if each["sig_proc"] == sig_proc:
        optim_result = each
        continue
    print(optim_result["sig_proc"])

    with open(os.path.join(args.outdir, "Interpolation_Check", sig_proc, "model.json"), "r") as f:
      interp_model = json.load(f)

    tagged_df = tagSignals(df, optim_result, proc_dict, [[mx, my]])

    for year in tagged_df.year.unique():
      for SR in tagged_df.SR.unique():
        print(year, SR)

        if args.fit_range_my:
          fit_range = [my-10*(my/100),my+10*(my/100)] #if y->gg
        else:
          fit_range = [112.5, 137.5] #everything else

        s = (tagged_df.year==year)&(tagged_df.SR==SR)
        if s.sum() < 100:
          print("Not plotting because not enough events")
          continue

        bin_centers, sumw, errors = histogram(tagged_df[s], fit_range, 50)

        popt_nom = nominal_model[str(year)][str(SR)]["%d_%d"%(mx, my)]["this mass"]["parameters"]
        popt_interp = interp_model[str(year)][str(SR)]["%d_%d"%(mx, my)]["this mass"]["parameters"]
        if not args.fit_range_my:
          mgg_peak = 125.0
        else:
          mgg_peak = my
        popt_nom[1] += mgg_peak
        popt_interp[1] += mgg_peak

        savedir = os.path.join(args.outdir, "Interpolation_Check", sig_proc, str(year), str(SR))
        os.makedirs(savedir, exist_ok=True)
        plotFitComparison(bin_centers, sumw, errors, fit_range, popt_nom, popt_interp, os.path.join(savedir, "interpolation_check_%d_%d_%d_%d.png"%(mx, my, year, SR)))



def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=True)
    return True

  df = pd.read_parquet(args.parquet_input)
  df = df[df.y==1]
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  common.add_MX_MY(df, proc_dict)
   
  with open(args.optim_results) as f:
     optim_results = json.load(f)

  plotChecks(df, proc_dict, optim_results, args)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--fit-range-my', action="store_true", help="Change fit range depending on my (for Y->gg)")
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)

      