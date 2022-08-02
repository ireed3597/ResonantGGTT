from multiprocessing import process
from unicodedata import numeric
import pandas as pd
import numpy as np
from xgboost import plot_importance, train
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
import torch

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
from plotting.training_plots import plotLoss

import common
import models
import preprocessing

import ast
import itertools

import tracemalloc
get_memory = lambda: np.array(tracemalloc.get_traced_memory())/1024**3
import copy
import pickle

from scipy.interpolate import interp1d

def loadDataFrame(args, train_features):
  columns_to_load = ["Diphoton_mass", "weight_central", "process_id", "category", "event", "year"] + train_features
  if not args.parquetSystematic: columns_to_load += common.weights_systematics
  columns_to_load = set(columns_to_load)

  print(">> Loading dataframe")
  df = pd.read_parquet(args.parquet_input, columns=columns_to_load)
  if args.dataset_fraction != 1.0:
    df = df.sample(frac=args.dataset_fraction)
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
    if i in needed_ids: print("> %s"%(reversed_proc_dict[i]).ljust(40), "kept")
    else: print("> %s"%(reversed_proc_dict[i]).ljust(40), "removed")
  df = df[df.process_id.isin(needed_ids)] #drop uneeded processes

  df["y"] = 0
  df.loc[df.process_id.isin(sig_ids), "y"] = 1

  return df, proc_dict

def train_test_split_consistent(MC, test_size, random_state):
  """
  Must ensure that for a given seed, the same splitting occurs for each process.
  If you do not consider this, then the splitting for mx=300 will be different if mx=500
  is left out of training for example.
  """
  train_dfs = []
  test_dfs = []

  idx = np.arange(len(MC))
  for proc in MC.process_id.unique():
    train_df, test_df = train_test_split(idx[MC.process_id==proc], test_size=test_size, random_state=random_state)
    train_dfs.append(train_df)
    test_dfs.append(test_df)

  return MC.iloc[np.concatenate(train_dfs)], MC.iloc[np.concatenate(test_dfs)]

  # for proc in MC.process_id.unique():
  #   train_df, test_df = train_test_split(MC[MC.process_id==proc], test_size=test_size, random_state=random_state)
  #   train_dfs.append(train_df)
  #   test_dfs.append(test_df)
  # return pd.concat(train_dfs), pd.concat(test_dfs)

def cv_fold_consistent(args, train_df, test_df):
  train_dfs = []
  test_dfs = []

  for proc in train_df.process_id.unique():
    proc_df = train_df[train_df.process_id==proc]

    fold_i, n_folds = args.cv_fold.split("/")
    fold_i, n_folds = int(fold_i), int(n_folds)
    kf = KFold(n_splits=n_folds)
    train_idx, test_idx = [each for each in kf.split(np.arange(len(proc_df)))][fold_i-1]
    
    train_proc_df, test_proc_df = proc_df.iloc[train_idx], proc_df.iloc[test_idx]

    train_dfs.append(train_proc_df)
    test_dfs.append(test_proc_df)

  return pd.concat(train_dfs), pd.concat(test_dfs)

#should work with BDT or NN
# def addScores(args, model, train_features, train_df, test_df, data, MX_MY_to_eval=None):
#   pd.options.mode.chained_assignment = None

#   if not args.parquetSystematic: dfs = [train_df, test_df, data]
#   else:                          dfs = [train_df, test_df]

#   #evaluate at nominal mass points
#   if MX_MY_to_eval is None:
#     MX_MY_to_eval = []
#     for sig_proc in args.eval_sig_procs:
#       MX, MY = common.get_MX_MY(sig_proc)
#       MX_MY_to_eval.append([MX, MY])

#   for MX, MY in MX_MY_to_eval:
#     #sig_proc = "XToHHggTauTau_M%d"%MX
#     sig_proc = "NMSSM_XYH_Y_gg_H_tautau_MX_%d_MY_%d"%(MX, MY)
#     print(sig_proc, MX, MY)
#     for df in dfs:
#       df.loc[:, "MX"] = MX
#       df.loc[:, "MY"] = MY
#       df["score_%s"%sig_proc] = model.predict_proba(df[train_features])[:,1]
      
#       #df.loc[:, "score_%s"%sig_proc] = (df["score_%s"%sig_proc] - df["score_%s"%sig_proc].min()) #rescale so everything within 0 and 1
#       #df.loc[:, "score_%s"%sig_proc] = (df["score_%s"%sig_proc] / df["score_%s"%sig_proc].max())
#       assert (df["score_%s"%sig_proc] < 0).sum() == 0
#       assert (df["score_%s"%sig_proc] > 1).sum() == 0

#   pd.options.mode.chained_assignment = "warn"

#specific to the paramNN, will not work with BDT
def addScores(args, model, train_features, train_df, test_df, data, MX_MY_to_eval=None):
  print(">> Adding scores (evaluating)")
  pd.options.mode.chained_assignment = None

  if not args.parquetSystematic: dfs = [train_df, test_df, data]
  else:                          dfs = [train_df, test_df]

  #evaluate at nominal mass points
  if MX_MY_to_eval is None:
    MX_MY_to_eval = []
    for sig_proc in args.eval_sig_procs:
      MX, MY = common.get_MX_MY(sig_proc)
      MX_MY_to_eval.append([MX, MY])

  for df in dfs:
    df.loc[:, "MX"] = MX_MY_to_eval[0][0]
    df.loc[:, "MY"] = MX_MY_to_eval[0][1]
  for i, (MX, MY) in enumerate(MX_MY_to_eval):
    for df in dfs:
      df.iloc[i, df.columns.get_loc("MX")] = MX
      df.iloc[i, df.columns.get_loc("MY")] = MY

  for df in dfs:
    all_predictions = model.predict_proba(df[train_features])
    for i, (MX, MY) in enumerate(MX_MY_to_eval):
      #sig_proc = "NMSSM_XYH_Y_gg_H_tautau_MX_%d_MY_%d"%(MX, MY)
      sig_proc = "XToHHggTauTau_M%d"%MX
      print(">", sig_proc, MX, MY)
      df["score_%s"%sig_proc] = all_predictions[i][:,1]
      assert (df["score_%s"%sig_proc] < 0).sum() == 0
      assert (df["score_%s"%sig_proc] > 1).sum() == 0

  pd.options.mode.chained_assignment = "warn"

def addTransformedScores(args, df, bkg):
  """
  Transforms scores such that bkg is flat.
  """

  for score_name in filter(lambda x: x.split("_")[0]=="score", df.columns):
    print(">", score_name)
    sig_proc = "_".join(score_name.split("_")[1:])
    intermediate_name = "intermediate_transformed_score_%s"%sig_proc

    bkg_score = bkg[score_name].to_numpy()
    bkg_weight = bkg["weight"].to_numpy()
    s = np.argsort(bkg_score)
    bkg_weight = bkg_weight[s]
    bkg_score = bkg_score[s]

    #only use pos
    bkg_score = bkg_score[bkg_weight>0]
    bkg_weight = bkg_weight[bkg_weight>0]

    bkg_cdf = np.cumsum(bkg_weight)
    bkg_cdf = bkg_cdf / bkg_cdf[-1]
    assert bkg_cdf[-1] == 1.0

    f32 = lambda x: np.array([x], dtype="float32")
    bkg_score = np.concatenate((f32(0.0), bkg_score, f32(1.0)))
    bkg_cdf = np.concatenate((f32(0.0), bkg_cdf, f32(1.0)))

    bkg_cdf_spline = interp1d(bkg_score, bkg_cdf, kind='linear')
    # x = np.linspace(bkg_score[np.argmin(abs(0.999-bkg_cdf))],  bkg_score[np.argmin(abs(1.0-bkg_cdf))], 100)
    # plt.clf()
    # plt.plot(x, bkg_cdf_spline(x))
    # plt.savefig("cdf/cdf_%s.png"%intermediate_name)
    # plt.clf()
    #assert np.isclose(bkg_cdf_spline(1.0), 1.0, atol=1e-5), print(bkg_cdf_spline(1.0))

    df[intermediate_name] = bkg_cdf_spline(df[score_name])

    assert (df[intermediate_name] < 0).sum() == 0
    assert (df[intermediate_name] > 1).sum() == 0, print(df[df[intermediate_name] > 1])

def doROC(args, train_df, test_df, sig_proc, proc_dict):
  #select just bkg and sig_proc
  train_df = train_df[(train_df.y==0)|(train_df.process_id==proc_dict[sig_proc])]
  test_df = test_df[(test_df.y==0)|(test_df.process_id==proc_dict[sig_proc])]

  if args.remove_gjets:
    gjet_ids = [proc_dict[proc] for proc in common.bkg_procs["GJets"]] + [proc_dict["TTJets"]]
    train_df = train_df[~train_df.process_id.isin(gjet_ids)]
    test_df = test_df[~test_df.process_id.isin(gjet_ids)]

  train_fpr, train_tpr, t = roc_curve(train_df.y, train_df["score_%s"%sig_proc], sample_weight=train_df.weight)
  test_fpr, test_tpr, t = roc_curve(test_df.y, test_df["score_%s"%sig_proc], sample_weight=test_df.weight)
  plotROC(train_fpr, train_tpr, test_fpr, test_tpr, os.path.join(args.outdir, sig_proc))

def importance_getter(model, X=None, y=None, w=None):
  model.importance_type = "gain"
  return model.feature_importances_

  # from sklearn.inspection import permutation_importance
  # print(sorted(list(X.columns)))
  # print(X.columns)
  # r = permutation_importance(model, X, y, sample_weight=w, n_repeats=5, random_state=0, n_jobs=-1)
  # print(r)
  # return r["importances_mean"]
  
def featureImportance(args, model, train_features, X=None, y=None, w=None):
  f, ax = plt.subplots(constrained_layout=True)
  f.set_size_inches(10, 20)

  plot_importance(model, ax)
  plt.savefig(os.path.join(args.outdir, args.train_sig_procs[0], "feature_importance.png"))
  plt.close()

  feature_importances = pd.Series(importance_getter(model, X, y, w), index=train_features)
  feature_importances.sort_values(ascending=False, inplace=True)
  print(feature_importances)
  with open(os.path.join(args.outdir, args.train_sig_procs[0], "feature_importances.json"), "w") as f:
    json.dump([feature_importances.to_dict(), feature_importances.index.to_list()], f, indent=4)

def findMassOrdering(args, model, train_df):
  """Find out order of sig procs in the train and test loss arrays from NN training"""
  sig_proc_ordering = ["" for proc in args.train_sig_procs]
  for proc in args.train_sig_procs:
    MX, MY = common.get_MX_MY(proc)
    dummy_X = train_df.iloc[0:1].copy()
    dummy_X.loc[:, "MX"] = MX
    dummy_X.loc[:, "MY"] = MY
    
    trans_dummy_X = model["transformer"].transform(dummy_X)
    for i, mass in enumerate(model["classifier"].mass_key):
      if abs(trans_dummy_X[0,-2:] - mass).sum() < 1e-4: #if found mass match
        sig_proc_ordering[i] = proc
        break
  return sig_proc_ordering

def evaluatePlotAndSave(args, proc_dict, model, train_features, train_df, test_df, data):
  models.setSeed(args.seed)
  print("Before adding scores", get_memory())
  addScores(args, model, train_features, train_df, test_df, data)
  print("After adding scores", get_memory())

  if not args.skipPlots:
    print(">> Plotting ROC curves")
    for sig_proc in args.eval_sig_procs:
      print(">", sig_proc)
      doROC(args, train_df, test_df, sig_proc, proc_dict)

    if hasattr(model["classifier"], "train_loss"):
      print(">> Plotting loss curves")
      train_loss = model["classifier"].train_loss
      validation_loss = model["classifier"].validation_loss
      plotLoss(train_loss.sum(axis=1), validation_loss.sum(axis=1), args.outdir)
      for i, proc in enumerate(findMassOrdering(args, model, train_df)):
        plotLoss(train_loss[:,i], validation_loss[:,i], os.path.join(args.outdir, proc))
  
  if args.only_ROC: return None
  
  #MX_MY_to_eval = [[mx, my] for mx in [300,400,500,600,700,800,900,1000] for my in [75,85,95]]
  #MX_MY_to_eval += [[mx, my] for mx in [350] for my in [70,75,80,85,90,95,100,125]]
  MX_MY_to_eval = [[mx,125] for mx in np.arange(260,1000,10)]
  addScores(args, model, train_features, train_df, test_df, data, MX_MY_to_eval)

  print(">> Creating output dataframe")
  if args.outputOnlyTest:
    output_df = pd.concat([test_df, data])
    output_df.loc[output_df.process_id!=proc_dict["Data"], "weight"] /= args.test_size #scale signal by amount thrown away
  else:
    output_df = pd.concat([test_df, train_df, data])
  del train_df, test_df, data
  output_bkg_MC = output_df[(output_df.y==0) & (output_df.process_id != proc_dict["Data"])]

  if args.loadTransformBkg is not None:
    print(">> Loading dataframe for score transformation")
    columns = list(filter(lambda x: "score" in x, output_df.columns)) + ["weight", "y", "process_id"]
    transform_df = pd.read_parquet(args.loadTransformBkg, columns=columns)
    transform_bkg = transform_df[(transform_df.y==0) & (transform_df.process_id != proc_dict["Data"])]
  else:
    transform_bkg = output_bkg_MC

  if args.remove_gjets:
    print(">> Removing gjet from transformation of score")
    gjet_ids = [proc_dict[proc] for proc in common.bkg_procs["GJets"]] + [proc_dict["TTJets"]]
    transform_bkg = transform_bkg[~transform_bkg.process_id.isin(gjet_ids)]

  print(">> Transforming scores")
  print("Before transforming", get_memory())
  addTransformedScores(args, output_df, transform_bkg)
  print("After transforming", get_memory())
  output_data = output_df[output_df.process_id == proc_dict["Data"]]
  output_bkg_MC = output_df[(output_df.y==0) & (output_df.process_id != proc_dict["Data"])]
  
  if not args.skipPlots:
    print(">> Plotting output scores")
    for sig_proc in args.eval_sig_procs:
      print(sig_proc)
      output_sig = output_df[output_df.process_id == proc_dict[sig_proc]]
      with np.errstate(divide='ignore', invalid='ignore'): plotOutputScore(output_data, output_sig, output_bkg_MC, proc_dict, sig_proc, os.path.join(args.outdir, sig_proc))

  columns_to_keep = ["Diphoton_mass", "weight", "process_id", "category", "event", "year", "y"]
  if not args.parquetSystematic: columns_to_keep += common.weights_systematics
  for column in output_df:
    if "score" in column: columns_to_keep.append(column)
  columns_to_keep = set(columns_to_keep)
  print(">> Outputting parquet file")
  output_df[columns_to_keep].to_parquet(os.path.join(args.outdir, args.outputName))

def main(args):
  os.makedirs(args.outdir, exist_ok=True)
  for sig_proc in args.eval_sig_procs:
    os.makedirs(os.path.join(args.outdir, sig_proc), exist_ok=True)

  models.setSeed(args.seed)

  train_features = common.train_features[args.train_features].copy()
  if "Param" in args.model: train_features += ["MX", "MY"]
  print(train_features)

  print("Before loading", get_memory())
  df, proc_dict = loadDataFrame(args, train_features)

  if args.feature_importance:
    train_features += ["random"]
    df["random"] = np.random.random(size=len(df))

  #shuffle bkg masses
  if "Param" in args.model:
    s = (df.y==0)&(df.process_id!=proc_dict["Data"])
    df.loc[s, "MX"] = np.random.choice(np.unique(df.loc[df.y==1, "MX"]), size=sum(s))
    df.loc[s, "MY"] = np.random.choice(np.unique(df.loc[df.y==1, "MY"]), size=sum(s))
  print("After loading", get_memory())

  MC = df.loc[~(df.process_id==proc_dict["Data"])]
  data = df.loc[df.process_id==proc_dict["Data"]]
  del df
  print("After half splitting", get_memory())


  train_df, test_df = train_test_split_consistent(MC, test_size=args.test_size, random_state=1)
  del MC
  train_df = train_df[train_df.weight>0]
  if args.data_as_bkg:
    train_df = pd.concat([train_df[train_df.y==1], data])
  if args.cv_fold is not None:
    train_df, test_df = cv_fold_consistent(args, train_df, test_df)
  print("After splitting", get_memory())

  if args.remove_gjets:
    gjet_ids = [proc_dict[proc] for proc in common.bkg_procs["GJets"]] + [proc_dict["TTJets"]]
    train_df = train_df[~train_df.process_id.isin(gjet_ids)]

  if not args.loadModel:
    if "Param" in args.model: classifier = getattr(models, args.model)(n_params=2, n_sig_procs=len(args.train_sig_procs), n_features=preprocessing.getNTransformedFeatures(train_df, train_features), hyperparams=args.hyperparams)
    else:                     classifier = getattr(models, args.model)(args.hyperparams)

    if args.drop_preprocessing:
      to_numpy = preprocessing.FunctionTransformer(lambda X, y=None: X.to_numpy())
      model = Pipeline([('to_numpy', to_numpy), ('classifier', classifier)])
      #model = Pipeline([('classifier', classifier)])
    else:
      numeric_features, categorical_features = preprocessing.autoDetermineFeatureTypes(train_df, train_features)
      print("Numeric features:", numeric_features)
      print("Categorical features:", categorical_features)
      model = Pipeline([('transformer', preprocessing.Transformer(numeric_features, categorical_features)), ('classifier', classifier)])

    sumw_before = train_df.weight.sum()
    print("Before training", get_memory())

    train_sig_ids = [proc_dict[sig_proc] for sig_proc in args.train_sig_procs]
    s = train_df.y==0 | train_df.process_id.isin(train_sig_ids)
    fit_params = {"classifier__w": train_df[s]["weight"]}
    if not args.drop_preprocessing: fit_params["transformer__w"] = train_df[s]["weight"]
    if hasattr(model["classifier"], "setOutdir"): model["classifier"].setOutdir(args.outdir)
    print(">> Training")
    model.fit(train_df[s][train_features], train_df[s]["y"], **fit_params)
    print(">> Training complete")
    print("After training", get_memory())

    assert sumw_before == train_df.weight.sum()

    if args.outputModel is not None:
      with open(args.outputModel, "wb") as f:
        pickle.dump(model, f)
  else:
    with open(args.loadModel, "rb") as f:
      model = pickle.load(f)

  if args.feature_importance:
    featureImportance(args, classifier.model, train_features, train_df[s][train_features], train_df[s]["y"], train_df[s]["weight"])

  evaluatePlotAndSave(args, proc_dict, model, train_features, train_df, test_df, data)

def expandSigProcs(sig_procs):
  expanded_sig_procs = []
  for sig_proc in sig_procs:
    expanded_sig_procs.extend(list(filter(lambda string: fnmatch.fnmatch(string, sig_proc), common.sig_procs["all"])))
  return expanded_sig_procs

def doParamTests(parser, args):
  args.do_param_tests = False
  
  #training on all
  args_copy = copy.deepcopy(args)
  args_copy.outdir = os.path.join(args.outdir, "all")
  start(parser, common.parserToList(args_copy))

  #training on individual
  if not args.skip_only_test:
    for sig_proc in args.train_sig_procs:
      args_copy = copy.deepcopy(args)
      args_copy.outdir = os.path.join(args.outdir, "only")
      args_copy.train_sig_procs = [sig_proc]
      args_copy.eval_sig_procs = [sig_proc]
      start(parser, common.parserToList(args_copy))

  #skip one
  for sig_proc in args.train_sig_procs:
    args_copy = copy.deepcopy(args)
    args_copy.outdir = os.path.join(args.outdir, "skip")
    args_copy.train_sig_procs.remove(sig_proc)
    args_copy.eval_sig_procs = [sig_proc]
    start(parser, common.parserToList(args_copy))


def doHyperParamSearch(parser, args):
  with open(args.hyperparams_grid, "r") as f:
    grid = json.load(f)
  args.hyperparams_grid = None

  original_outdir = args.outdir

  keys, values = zip(*grid.items())
  experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
  for i, experiment in enumerate(experiments):
    args_copy = copy.deepcopy(args)

    args_copy.outdir = os.path.join(original_outdir, "experiment_%d"%i)
    os.makedirs(args_copy.outdir, exist_ok=True)

    hyperparams_path = os.path.join(args_copy.outdir, "hyperparameters.json")
    with open(hyperparams_path, "w") as f:
      json.dump(experiment, f, indent=4)
    args_copy.hyperparams = hyperparams_path

    start(parser, common.parserToList(args_copy))

def doCV(parser, args):
  original_outdir = args.outdir
  n_folds = args.do_cv
  
  args.do_cv = 0

  for i in range(1, n_folds+1):
    args_copy = copy.deepcopy(args)
    args_copy.outdir = os.path.join(original_outdir, "cv_fold_%d"%i)
    args_copy.cv_fold = "%d/%d"%(i, n_folds)

    start(parser, common.parserToList(args_copy))

def start(parser, args=None):
  args = parser.parse_args(args)

  if args.parquetSystematic: assert args.loadTransformBkg is not None

  if args.eval_sig_procs == None:
    args.eval_sig_procs = args.train_sig_procs
  args.train_sig_procs = expandSigProcs(args.train_sig_procs)
  args.train_sig_procs_exclude = expandSigProcs(args.train_sig_procs_exclude)
  args.eval_sig_procs = expandSigProcs(args.eval_sig_procs)
  
  args.train_sig_procs = list(filter(lambda x: x not in args.train_sig_procs_exclude, args.train_sig_procs))

  """
  if args.feature_importance:
    assert args.model == "BDT"
    assert len(args.train_sig_procs) == len(args.eval_sig_procs) == 1
    assert args.drop_preprocessing
  """

  if args.hyperparams_grid != None:
    assert args.hyperparams == None
    doHyperParamSearch(parser, args)
    return True

  if args.do_param_tests:
    #assert args.batch
    doParamTests(parser, args)
    return True

  if args.do_cv > 0:
    doCV(parser, args)
    return True
    
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
    return True

  if args.hyperparams != None:
    with open(args.hyperparams, "r") as f:
      args.hyperparams = json.load(f)
    print(args.hyperparams)

  print(">> Will train on:")
  print("\n>".join(args.train_sig_procs))
  print(">> Will evaluate on:")
  print("\n>".join(args.eval_sig_procs))

  tracemalloc.start()

  df = main(args)

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
  parser.add_argument('--data-as-bkg', action="store_true", help="Use data as training background")

  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)

  parser.add_argument('--feature-importance', action="store_true")
  parser.add_argument('--do-param-tests', action="store_true")
  parser.add_argument('--skip-only-test', action="store_true")
  parser.add_argument('--only-ROC', action="store_true")
  parser.add_argument('--remove-gjets', action="store_true")
  parser.add_argument('--dataset-fraction', type=float, default=1.0, help="Only use a fraction of the whole dataset.")

  parser.add_argument('--hyperparams',type=str, default=None)
  parser.add_argument('--hyperparams-grid', type=str, default=None)

  parser.add_argument('--do-cv', type=int, default=0, help="Give a non-zero number which specifies the number of folds to do for cv. Will then run script over all folds.")
  parser.add_argument('--cv-fold', type=str, default=None, help="If doing cross-validation, specify the number of folds and which to run on. Example: '--cv-fold 2/5' means the second out of five folds.")

  parser.add_argument('--outputModel', type=str, default=None)
  parser.add_argument('--loadModel', type=str, default=None)
  parser.add_argument('--outputName', type=str, default="output.parquet")
  parser.add_argument('--skipPlots', action="store_true")

  parser.add_argument('--parquetSystematic', action="store_true")
  parser.add_argument('--loadTransformBkg', type=str, default=None)

  #import cProfile
  #cProfile.run('start(parser)', 'restats')
  start(parser)