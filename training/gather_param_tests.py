import json
import os
import argparse
import pandas as pd
import tabulate
import common

def getAUCScore(path, sig_proc):
  if os.path.exists(os.path.join(path, sig_proc, "ROC_skimmed.json")):
    ROC_path = "ROC_skimmed.json"
  else:
    ROC_path = "ROC.json"

  with open(os.path.join(path, sig_proc, ROC_path), "r") as f:
    ROC = json.load(f)
  return ROC["test_auc"]

def getGoodnessScores(results):
  """
  Scores to test how well the parameterised model is performing.
  Need to test:
  - Overall performance
  - Closeness to single mass point performance
  - Interpolation ability (skipping a mass point)
  Also define a goodness score which is a combination of overall performance
  and interpolation ability.
  """
  scores = {}

  #overall performance (avg auc score across 'all')
  scores["avg_auc"] = results['all'].mean()

  #closeness
  scores["closeness"] = (results['all'] - results['only']).sum()

  #interpolation ability
  scores["interp"] = (results[1:-1]['skip'] - results[1:-1]['all']).sum()

  #goodness score
  scores["goodness"] = scores["avg_auc"] * (1+scores["interp"])

  return scores

def gatherExperimentResults(path):
  all_path = os.path.join(path, "all")
  only_path = os.path.join(path, "only")
  skip_path = os.path.join(path, "skip")
  if not os.path.exists(only_path): only_path = all_path

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
  
  with open(os.path.join(path, "param_test_results.txt"), "w") as f:
    f.write(table)
  with open(os.path.join(path, "param_test_results.tex"), "w") as f:
    f.write(results.to_latex(float_format="%.4f"))
  results.to_csv(os.path.join(path, "param_test_results.csv"), float_format="%.4f")

  return results

def getHyperParams(outdir, i):
  path = os.path.join(outdir, "experiment_%d"%i, "hyperparameters.json")
  print(path)
  if os.path.exists(path):
    with open(path, "r") as f:
      return json.load(f)
  else:
    return "No Hyperparameter file"

def findBest(results, name, args):
  best = [-1, None, {name:-9999}]
  for i, each in enumerate(results):
    table = tabulate.tabulate(each.T, headers='keys', floatfmt=".4f")
    scores = getGoodnessScores(each)

    if scores[name] > best[2][name]:
      best[0] = i
      best[1] = table
      best[2] = scores

  print("\n%s"%name)
  print("Experiment %d was best:"%best[0])
  print(best[1])
  print(best[2])
  print(getHyperParams(args.outdir, best[0]))

def main(args):
  if not os.path.exists(os.path.join(args.outdir, "experiment_0")):
    results = [gatherExperimentResults(args.outdir)]
  else:
    results = []
    for directory in os.listdir(args.outdir):
      #if "experiment" in directory and os.path.exists(os.path.join(args.outdir, directory, "all")):
      if "experiment" in directory and os.path.exists(os.path.join(args.outdir, directory, "all")):
        print(directory)
        results.append(gatherExperimentResults(os.path.join(args.outdir, directory)))

  for i, each in enumerate(results):
    table = tabulate.tabulate(each.T, headers='keys', floatfmt=".4f")
    scores = getGoodnessScores(each)
    print("Experiment %d:"%i)
    print(table)
    print(scores)
    print(getHyperParams(args.outdir, i))

  for name in ["avg_auc", "closeness", "interp", "goodness"]:
    findBest(results, name, args)

  return results

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--outdir', '-o', type=str, required=True)

  args = parser.parse_args()

  results = main(args)