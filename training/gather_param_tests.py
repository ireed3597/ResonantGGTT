import json
import os
import argparse
import pandas as pd
import tabulate
import common

def getAUCScore(path, sig_proc):
  with open(os.path.join(path, sig_proc, "ROC.json"), "r") as f:
    ROC = json.load(f)
  return ROC["test_auc"]

def main(args):
  all_path = os.path.join(args.outdir, "all")
  only_path = os.path.join(args.outdir, "only")
  skip_path = os.path.join(args.outdir, "skip")

  sig_procs = list(filter(lambda x: os.path.isdir(os.path.join(all_path, x)), os.listdir(all_path)))

  results = {}
  results["all"] = [getAUCScore(all_path, sig_proc) for sig_proc in sig_procs]
  results["only"] = [getAUCScore(only_path, sig_proc) for sig_proc in sig_procs]
  results["skip"] = [getAUCScore(skip_path, sig_proc) for sig_proc in sig_procs]

  results = pd.DataFrame(results, index=sig_procs)
  results["MX"] = 0
  results["MY"] = 0
  for sig_proc in sig_procs:
    MX, MY = common.get_MX_MY(sig_proc)
    results.loc[sig_proc, "MX"] = MX
    results.loc[sig_proc, "MY"] = MY
  results.sort_values("MX", inplace=True)

  nice_index = (["MX_%d_MY_%d"%(row.MX, row.MY) for index, row in results.iterrows()])
  results.index = nice_index

  results.drop(["MX", "MY"], axis=1, inplace=True)

  table = tabulate.tabulate(results.T, headers='keys', floatfmt=".4f")
  print(table)

  with open(os.path.join(args.outdir, "param_test_results.txt"), "w") as f:
    f.write(table)
  with open(os.path.join(args.outdir, "param_test_results.tex"), "w") as f:
    f.write(results.to_latex(float_format="%.4f"))
  results.to_csv(os.path.join(args.outdir, "param_test_results.csv"), float_format="%.4f")

  return results

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--outdir', '-o', type=str, required=True)

  args = parser.parse_args()

  results = main(args)