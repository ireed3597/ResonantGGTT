import pandas as pd
import numpy as np
from xgboost import plot_importance
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import argparse
import json
import fnmatch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys
from plotting.training_plots import plotOutputScore
from plotting.training_plots import plotROC

import common
import models
import preprocessing

import ast
import itertools

def loadDataFrame(args, train_features):
  columns_to_load = set(["Diphoton_mass", "weight_central", "process_id", "category", "event", "year"] + train_features)

  print(">> Loading dataframe")
  df = pd.read_parquet(args.parquet_input, columns=columns_to_load)
  df.rename({"weight_central": "weight"}, axis=1, inplace=True)
  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']

  sig_procs_to_keep = set(args.train_sig_procs + args.eval_sig_procs)

  sig_ids = [proc_dict[proc] for proc in sig_procs_to_keep]
  bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"] if proc in proc_dict.keys()]
  data_ids = [proc_dict["Data"]]
  needed_ids = sig_ids+bkg_ids+data_ids
  
  reversed_proc_dict = {proc_dict[key]:key for key in proc_dict.keys()}
  for i in df.process_id.unique():
    if i in needed_ids: print("> %s"%(reversed_proc_dict[i]).ljust(30), "kept")
    else: print("> %s"%(reversed_proc_dict[i]).ljust(30), "removed")
  df = df[df.process_id.isin(needed_ids)] #drop uneeded processes

  df["y"] = 0
  df.loc[df.process_id.isin(sig_ids), "y"] = 1

  return df, proc_dict

def addScores(args, model, train_features, train_df, test_df, data):
  pd.options.mode.chained_assignment = None

  dfs = [train_df, test_df, data]
  for sig_proc in args.eval_sig_procs:
    MX, MY = common.get_MX_MY(sig_proc)
    print(sig_proc, MX, MY)
    for df in dfs:
      df.loc[:, "MX"] = MX
      df.loc[:, "MY"] = MY
      df["score_%s"%sig_proc] = model.predict_proba(df[train_features])[:,1] + np.random.normal(scale=1e-8, size=len(df))

  pd.options.mode.chained_assignment = "warn"

def tan(x, b, c, d, f):
  e = (1/(1-f))*np.arctan(c/d)
  a = (e*d)/b
  return (a*np.tan((x-f)*b) - c)*(x<f) + (d*np.tan((x-f)*e) - c)*(x>=f)
popt = [1.3982465170462963, 2.1338810272238735, -0.2513888030857778, 0.7889447703857513] #nmssm
generic_sig_cdf = lambda x: np.power(10, tan(x, *popt)) / np.power(10, tan(1, *popt))

def addTransformedScores(args, df, bkg):
  """
  Transforms scores such that bkg is flat.
  """

  for sig_proc in args.eval_sig_procs:
    score_name = "score_%s"%sig_proc

    df.sort_values(score_name, inplace=True)
    bkg.sort_values(score_name, inplace=True)  
    
    df_score = df[score_name].to_numpy()
    bkg_score = bkg[score_name].to_numpy()
    bkg_cdf = (np.cumsum(bkg.weight)/np.sum(bkg.weight)).to_numpy()
    idx = np.searchsorted(bkg_score, df_score, side="right")
    idx[idx == len(bkg_cdf)] = len(bkg_cdf) - 1 #if df score > max(bkg_score) it will give an index out of bounds
    
    intermediate_name = "intermediate_transformed_score_%s"%sig_proc
    df[intermediate_name] = bkg_cdf[idx]
    df.loc[df[intermediate_name]<0, intermediate_name] = 0
    df.loc[df[intermediate_name]>1, intermediate_name] = 1

    transformed_name = "transformed_score_%s"%sig_proc
    df[transformed_name] = generic_sig_cdf(df[intermediate_name])
    df.loc[df[transformed_name]<0, transformed_name] = 0
    df.loc[df[transformed_name]>1, transformed_name] = 1
    
    assert (df[score_name] < 0).sum() == 0
    assert (df[score_name] > 1).sum() == 0
    assert (df[intermediate_name] < 0).sum() == 0
    assert (df[intermediate_name] > 1).sum() == 0
    assert (df[transformed_name] < 0).sum() == 0
    assert (df[transformed_name] > 1).sum() == 0

def doROC(train_df, test_df, sig_proc, proc_dict):
  #select just bkg and sig_proc
  train_df = train_df[(train_df.y==0)|(train_df.process_id==proc_dict[sig_proc])]
  test_df = test_df[(test_df.y==0)|(test_df.process_id==proc_dict[sig_proc])]

  train_fpr, train_tpr, t = roc_curve(train_df.y, train_df["score_%s"%sig_proc], sample_weight=train_df.weight)
  test_fpr, test_tpr, t = roc_curve(test_df.y, test_df["score_%s"%sig_proc], sample_weight=test_df.weight)
  plotROC(train_fpr, train_tpr, test_fpr, test_tpr, os.path.join(args.outdir, sig_proc))

def importance_getter(model):
  model.importance_type = "weight"
  return model.feature_importances_

def featureImportance(args, model, train_features):
  f, ax = plt.subplots(constrained_layout=True)
  f.set_size_inches(10, 20)

  plot_importance(model, ax)
  plt.savefig(os.path.join(args.outdir, args.train_sig_procs[0], "feature_importance.png"))
  plt.close()

  feature_importances = pd.Series(importance_getter(model), index=train_features)
  feature_importances.sort_values(ascending=False, inplace=True)
  print(feature_importances)
  with open(os.path.join(args.outdir, args.train_sig_procs[0], "feature_importances.json"), "w") as f:
    json.dump([feature_importances.to_dict(), feature_importances.index.to_list()], f, indent=4)

def evaluatePlotAndSave(args, proc_dict, model, train_features, train_df, test_df, data):
  addScores(args, model, train_features, train_df, test_df, data)

  print(">> Plotting ROC curves")
  for sig_proc in args.eval_sig_procs:
    print(sig_proc)
    doROC(train_df, test_df, sig_proc, proc_dict)

  if args.outputOnlyTest:
    output_df = pd.concat([test_df, data])
    output_df.loc[output_df.process_id!=proc_dict["Data"], "weight"] /= args.test_size #scale signal by amount thrown away
  else:
    output_df = pd.concat([test_df, train_df, data])

  output_bkg_MC = output_df[(output_df.y==0) & (output_df.process_id != proc_dict["Data"])]
  print(">> Transforming scores")
  addTransformedScores(args, output_df, output_bkg_MC)

  output_bkg_MC = output_df[(output_df.y==0) & (output_df.process_id != proc_dict["Data"])]
  output_data = output_df[output_df.process_id == proc_dict["Data"]]
  print(">> Plotting output scores")
  for sig_proc in args.eval_sig_procs:
    print(sig_proc)
    output_sig = output_df[output_df.process_id == proc_dict[sig_proc]]
    with np.errstate(divide='ignore', invalid='ignore'): plotOutputScore(output_data, output_sig, output_bkg_MC, proc_dict, sig_proc, os.path.join(args.outdir, sig_proc))

  columns_to_keep = ["Diphoton_mass", "weight", "process_id", "category", "event", "year", "y"]
  for column in output_df:
    if "score" in column: columns_to_keep.append(column)
  print(">> Outputting parquet file")
  output_df[columns_to_keep].to_parquet(os.path.join(args.outdir, "output.parquet"))

def main(args):
  os.makedirs(args.outdir, exist_ok=True)
  for sig_proc in args.eval_sig_procs:
    os.makedirs(os.path.join(args.outdir, sig_proc), exist_ok=True)

  models.setSeed(args.seed)

  train_features = common.train_features[args.train_features]
  if "Param" in args.model: train_features += ["MX", "MY"]
  print(train_features)

  print("Before loading", tracemalloc.get_traced_memory())
  df, proc_dict = loadDataFrame(args, train_features)
  
  print("After loading", tracemalloc.get_traced_memory())
  MC = df[~(df.process_id==proc_dict["Data"])]
  data = df[df.process_id==proc_dict["Data"]]
  del df

  train_df, test_df = train_test_split(MC, test_size=args.test_size)
  del MC
  print("After splitting", tracemalloc.get_traced_memory())

  train_features = common.train_features[args.train_features]
  train_sig_ids = [proc_dict[sig_proc] for sig_proc in args.train_sig_procs]

  if "Param" in args.model: classifier = getattr(models, args.model)(n_params=2, n_sig_procs=len(args.train_sig_procs))
  else:                     classifier = getattr(models, args.model)(args.hyperparams)

  if args.drop_preprocessing:
    to_numpy = preprocessing.FunctionTransformer(lambda X, y=None: X.to_numpy())
    model = Pipeline([('to_numpy', to_numpy), ('classifier', classifier)])
  else:
    numeric_features, categorical_features = preprocessing.autoDetermineFeatureTypes(train_df, train_features)
    model = Pipeline([('transformer', preprocessing.Transformer(numeric_features, categorical_features)), ('classifier', classifier)])

  sumw_before = train_df.weight.sum()

  print("Before training", tracemalloc.get_traced_memory())

  s = train_df.y==0 | train_df.process_id.isin(train_sig_ids)
  fit_params = {"classifier__w": train_df[s]["weight"]}
  if not args.drop_preprocessing: fit_params["transformer__w"] = train_df[s]["weight"]
  print(">> Training")
  model.fit(train_df[s][train_features], train_df[s]["y"], **fit_params)
  print(">> Training complete")

  print("After training", tracemalloc.get_traced_memory())

  assert sumw_before == train_df.weight.sum()

  if args.feature_importance:
    featureImportance(args, classifier.model, train_features)

  evaluatePlotAndSave(args, proc_dict, model, train_features, train_df, test_df, data)

def expandSigProcs(sig_procs):
  expanded_sig_procs = []
  for sig_proc in sig_procs:
    expanded_sig_procs.extend(list(filter(lambda string: fnmatch.fnmatch(string, sig_proc), common.sig_procs["all"])))
  return expanded_sig_procs

def doParamTests(args):
  args.do_param_tests = False
  
  #training on all
  import copy
  args_copy = copy.deepcopy(args)
  args_copy.outdir = os.path.join(args.outdir, "all")
  common.submitToBatch([sys.argv[0]] + common.parserToList(args_copy))

  #training on individual
  for sig_proc in args.train_sig_procs:
    args_copy = copy.deepcopy(args)
    args_copy.outdir = os.path.join(args.outdir, "only")
    args_copy.train_sig_procs = [sig_proc]
    args_copy.eval_sig_procs = [sig_proc]
    common.submitToBatch([sys.argv[0]] + common.parserToList(args_copy))

  #skip one
  #training on individual
  for sig_proc in args.train_sig_procs:
    args_copy = copy.deepcopy(args)
    args_copy.outdir = os.path.join(args.outdir, "skip")
    args_copy.train_sig_procs.remove(sig_proc)
    args_copy.eval_sig_procs = [sig_proc]
    common.submitToBatch([sys.argv[0]] + common.parserToList(args_copy))

def doHyperParamSearch(args):
  with open(args.hyperparams_grid, "r") as f:
    grid = json.load(f)
  args.hyperparams_grid = None

  original_outdir = args.outdir

  args.hyperparams_grid == None
  keys, values = zip(*grid.items())
  experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
  for i, experiment in enumerate(experiments):
    args.outdir = os.path.join(original_outdir, "experiment_%d"%i)
    os.makedirs(args.outdir, exist_ok=True)

    hyperparams_path = os.path.join(args.outdir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
      json.dump(experiment, f, indent=4)
    args.hyperparams = hyperparams_path
    command = "python %s %s"%(sys.argv[0], " ".join(common.parserToList(args)))
    print(command)
    os.system(command)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--train-sig-procs', '-p', type=str, nargs="+", required=True)
  parser.add_argument('--train-sig-procs-exclude', type=str, nargs="+", default=[])
  parser.add_argument('--eval-sig-procs', type=str, nargs="+")
  parser.add_argument('--train-features', type=str, default="all")
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--model', type=str, default="BDT")
  parser.add_argument('--outputOnlyTest', action="store_true", default=False)
  parser.add_argument('--test-size', type=float, default=0.5)
  parser.add_argument('--drop-preprocessing', action="store_true")
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--feature-importance', action="store_true")
  parser.add_argument('--do-param-tests', action="store_true")

  parser.add_argument('--hyperparams',type=str, default=None)
  parser.add_argument('--hyperparams-grid', type=str, default=None)

  args = parser.parse_args()

  if args.eval_sig_procs == None:
    args.eval_sig_procs = args.train_sig_procs
  args.train_sig_procs = expandSigProcs(args.train_sig_procs)
  args.train_sig_procs_exclude = expandSigProcs(args.train_sig_procs_exclude)
  args.eval_sig_procs = expandSigProcs(args.eval_sig_procs)
  
  args.train_sig_procs = list(filter(lambda x: x not in args.train_sig_procs_exclude, args.train_sig_procs))

  if args.feature_importance:
    assert args.model == "BDT"
    assert len(args.train_sig_procs) == len(args.eval_sig_procs) == 1
    assert args.drop_preprocessing

  if args.hyperparams_grid != None:
    assert args.hyperparams == None
    doHyperParamSearch(args)
    exit(0)

  if args.hyperparams != None:
    with open(args.hyperparams, "r") as f:
      args.hyperparams = json.load(f)
    print(args.hyperparams)

  if args.do_param_tests:
    assert args.batch
    doParamTests(args)
    exit(0)

  if args.batch:
    common.submitToBatch(sys.argv)
    exit(0)

  print(">> Will train on:")
  print("\n".join(args.train_sig_procs))
  print(">> Will evaluate on:")
  print("\n".join(args.eval_sig_procs))

  import tracemalloc
  tracemalloc.start()

  df = main(args)