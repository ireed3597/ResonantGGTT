from builtins import sorted
import pandas as pd
import numpy as np
import matplotlib

import signalModelling.systematics as syst
syst.Y_gg = False # so systematics scripts what ranges to use for mgg
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

import copy

mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

def fitSignalModel(df, outdir, savename="single_higgs_bkg", make_plots=False):  
  if make_plots: 
    os.makedirs(outdir, exist_ok=True)
    popt, perr = fitDCB(df, fit_range=[125-12.5,125+12.5], savepath=os.path.join(outdir, "%s.png"%savename))
  else:          
    popt, perr = fitDCB(df, fit_range=[125-12.5,125+12.5], savepath=None)

  popt[1] = popt[1] - 125.0 #change mean to a delta (dm)

  return popt, perr

def tagSignals(df, entry):
  pd.options.mode.chained_assignment = None

  df.loc[:, "SR"] = -1

  boundaries = entry["category_boundaries"][::-1]

  score = entry["score"]
  if score not in df.columns: #intermediate mass points not in systematics
    return df.iloc[0:0]

  for i in range(len(boundaries)-1):
    selection = (df[score] <= boundaries[i]) & (df[score] > boundaries[i+1])
    df.loc[selection, "SR"] = i

  pd.options.mode.chained_assignment = "warn"
  return df[df.SR!=-1]

def getClosest(mass, all_masses):
  mass = np.array(mass.split("_"), dtype=float)
  all_masses = np.array(all_masses, dtype=float)
  r = np.sqrt((mass[0]-all_masses[:,0])**2 + (mass[0]-all_masses[:,0])**2)
  sorted_masses = all_masses[np.argsort(r)]
  return ["%d_%d"%tuple(m) for m in sorted_masses]

def findSuccessfulSystematics(systematics):
  for k1, v1 in systematics.items():            # year
    for k2, v2 in systematics[k1].items():      # SR
      for k3, v3 in systematics[k1][k2].items():    # mass
        for k4, v4 in systematics[k1][k2][k3].items():  # proc
          if type(v4) is dict:
            return v4

def deriveModels(dfs, proc_dict, optim_results, original_outdir, make_plots=False, systematics=False):
  nSR = len(optim_results[0]["category_boundaries"]) - 1
  models = {str(year):{str(SR):{} for SR in range(nSR)} for year in np.unique(dfs["nominal"].year)}

  if systematics:
    systematics = {str(year):{str(SR):{} for SR in range(nSR)} for year in np.unique(dfs["nominal"].year)}
    yield_systematics = syst.getYieldSystematicsNames(dfs["nominal"])
    parquet_yield_systematics = ["JER", "JES", "MET_JER", "MET_JES", "MET_Unclustered", "Muon_pt", "Tau_pt"]
    parquet_shape_systematics = ["fnuf", "material", "scale", "smear"]
    #parquet_yield_systematics = ["JER"]
    #parquet_shape_systematics = ["fnuf"]

  for year in dfs["nominal"].year.unique():
  #for year in [2016]:
    print(year)
    for entry in optim_results:
      MX, MY = common.get_MX_MY(entry["sig_proc"])
      if MY != 90: continue
      mass = "%d_%d"%(MX, MY)
      print(mass)
      for key,df in dfs.items():
        df.loc[:, "MX"] = MX
        df.loc[:, "MY"] = MY
      
      dfs_tagged = {}
      for key in dfs.keys():
        dfs_tagged[key] = tagSignals(dfs[key][dfs[key].year==year], entry)
      
      for SR in dfs_tagged["nominal"].SR.unique():
      #for SR in [0, 5]:
        outdir = os.path.join(original_outdir, str(year), str(SR), "%d_%d"%(MX,MY))
        
        dfs_SR = {}
        for key in dfs.keys():
          dfs_SR[key] = dfs_tagged[key][dfs_tagged[key].SR==SR]

        models[str(year)][str(SR)][mass] = {}
        if systematics: systematics[str(year)][str(SR)][mass] = {}
        
        for proc in common.bkg_procs["SM Higgs"]:
          dfs_proc = {}
          for key in dfs.keys():
            dfs_proc[key] = dfs_SR[key][dfs_SR[key].process_id==proc_dict[proc]]
          
          df_proc = dfs_proc["nominal"]

          if (df_proc.weight.sum() > 0.1): #if resonant bkg contribute enough            
            if (len(df_proc) > 1000) & (df_proc.weight.sum() > 0.1):
              popt, perr = fitSignalModel(df_proc, outdir, savename=proc, make_plots=make_plots)
              
              if systematics:
                #only do systematics if nominal mass
                if len(dfs_SR["%s_up"%parquet_yield_systematics[0]]) > 0:                  
                  systematics[str(year)][str(SR)][mass][proc] = {}
                  for systematic in yield_systematics:
                    systematics[str(year)][str(SR)][mass][proc][systematic] = syst.deriveYieldSystematic(dfs_SR["nominal"], systematic, mass)
                  for systematic in parquet_yield_systematics:
                    systematics[str(year)][str(SR)][mass][proc][systematic] = syst.deriveParquetYieldSystematic(dfs_SR, systematic, mass)
                  for systematic in parquet_shape_systematics:
                    systematics[str(year)][str(SR)][mass][proc].update(syst.deriveParquetShapeSystematics(dfs_SR, systematic, mass))
                else:
                  systematics[str(year)][str(SR)][mass][proc] = "closest"

            else:
              popt, perr = [], []
              if systematics:
                if proc == "ttH_M125":
                  systematics[str(year)][str(SR)][mass][proc] = "no systematics"
                else:
                  systematics[str(year)][str(SR)][mass][proc] = "from ttH"
            norm = df_proc.weight.sum() / (1000 * common.lumi_table[year]) #finalFits expects picobarn
          
          else: #if don't need a model
            popt = np.array([1, 124.8-125, 1.35, 1.3, 4.5, 2.1, 2.8]) #from a ttH model with good stats
            perr = np.zeros_like(popt)
            norm = 0

            if systematics:
              systematics[str(year)][str(SR)][mass][proc] = "no systematics"
          
          models[str(year)][str(SR)][mass][proc] = {}
          models[str(year)][str(SR)][mass][proc]["parameters"] = list(popt)
          models[str(year)][str(SR)][mass][proc]["parameters_err"] = list(perr)
          models[str(year)][str(SR)][mass][proc]["norm"] = norm

        #fill in missing models
        for proc in common.bkg_procs["SM Higgs"]:
          if (models[str(year)][str(SR)][mass][proc]["parameters"] == []) & (models[str(year)][str(SR)][mass][proc]["norm"] != 0): #if couldn't derive model
            
            # check ttH model exists, if not, give resonable shape parameters to ttH
            if models[str(year)][str(SR)][mass]["ttH_M125"]["parameters"] == []:
              models[str(year)][str(SR)][mass]["ttH_M125"]["parameters"] = np.array([1, 124.8-125, 1.35, 1.3, 4.5, 2.1, 2.8])
              models[str(year)][str(SR)][mass][proc]["parameters_err"] = np.zeros_like(popt)

            models[str(year)][str(SR)][mass][proc]["parameters"] = models[str(year)][str(SR)][mass]["ttH_M125"]["parameters"]
            models[str(year)][str(SR)][mass][proc]["parameters_err"] = models[str(year)][str(SR)][mass]["ttH_M125"]["parameters_err"]

    if systematics:
      for SR in dfs_tagged["nominal"].SR.unique():
        all_masses = []
        for mass in systematics[str(year)][str(SR)]:
          all_masses.append(mass.split("_"))

        #put in no systematics
        no_sys_dict = copy.deepcopy(findSuccessfulSystematics(systematics))
        for key in no_sys_dict.keys():
          if "const" in key:
            no_sys_dict[key] = 0.0
          else:
            no_sys_dict[key] = 1.0
        for mass in systematics[str(year)][str(SR)]:
          for proc in common.bkg_procs["SM Higgs"]:
            if systematics[str(year)][str(SR)][mass][proc] == "no systematics":
              systematics[str(year)][str(SR)][mass][proc] = no_sys_dict

        #make ttH replacement
        for mass in systematics[str(year)][str(SR)]:
          for proc in common.bkg_procs["SM Higgs"]:
            if systematics[str(year)][str(SR)][mass][proc] == "from ttH":
              systematics[str(year)][str(SR)][mass][proc] = systematics[str(year)][str(SR)][mass]["ttH_M125"]

        #replace systematics with closest
        for mass in systematics[str(year)][str(SR)]:
          for proc in common.bkg_procs["SM Higgs"]:
            if systematics[str(year)][str(SR)][mass][proc] == "closest":
              for closest_mass in getClosest(mass, all_masses):
                if type(systematics[str(year)][str(SR)][closest_mass][proc]) is dict:
                  print(mass, closest_mass)
                  closest_systematics = systematics[str(year)][str(SR)][closest_mass][proc]
                  systematics[str(year)][str(SR)][mass][proc] = closest_systematics
                  break
              assert type(systematics[str(year)][str(SR)][mass][proc]) is dict

        

  #at this stage we have

  with open(os.path.join(original_outdir, "model.json"), "w") as f:
    json.dump(models, f, indent=4, sort_keys=True, cls=common.NumpyEncoder)
  with open(os.path.join(original_outdir, "systematics.json"), "w") as f:
    json.dump(systematics, f, indent=4, sort_keys=True, cls=common.NumpyEncoder)


def loadDataFrame(path, proc_dict, columns=None, sample_fraction=1.0):
  if columns is None:
    columns = common.getColumns(path)
    columns = list(filter(lambda x: x[:5] != "score", columns))
  df = pd.read_parquet(path, columns=columns)
  if sample_fraction != 1.0 :df = df.sample(frac=sample_fraction)
  df = df[df.process_id.isin([proc_dict[proc] for proc in common.bkg_procs["SM Higgs"]])]

  return df

def loadDataFrames(args, proc_dict):
  dfs = {}
  
  #load nominal dataframe
  df = loadDataFrame(os.path.join(args.parquet_input, "merged_nominal.parquet"), proc_dict, sample_fraction=args.dataset_fraction)
  #systematic_columns = list(filter(lambda x: ("intermediate_transformed_score" in x), df.columns)) + ["Diphoton_mass", "process_id", "weight", "y", "year"]
  dfs["nominal"] = df

  if args.systematics:
    for path in os.listdir(args.parquet_input):
      if (".parquet" in path) and ("nominal" not in path):
      #if "fnuf" in path or "merged_JER" in path:
        print(path)
        #df = loadDataFrame(os.path.join(args.parquet_input, path), proc_dict, columns=systematic_columns)
        df = loadDataFrame(os.path.join(args.parquet_input, path), proc_dict, sample_fraction=args.dataset_fraction)
        name = "_".join(path.split(".parquet")[0].split("_")[1:])
        dfs[name] = df

  return dfs

def main(args):
  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
    return True

  with open(args.summary_input) as f:
    proc_dict = json.load(f)['sample_id_map']
  with open(args.optim_results) as f:
     optim_results = json.load(f)

  dfs = loadDataFrames(args, proc_dict)

  deriveModels(dfs, proc_dict, optim_results, args.outdir, make_plots=args.make_plots, systematics=args.systematics)  


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--optim-results', '-r', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--outdir', '-o', type=str, required=True)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=2)
  parser.add_argument('--make-plots', action="store_true")
  parser.add_argument('--systematics', action="store_true")
  parser.add_argument('--dataset-fraction', type=float, default=1.0)
  args = parser.parse_args()
  
  os.makedirs(args.outdir, exist_ok=True)

  main(args)