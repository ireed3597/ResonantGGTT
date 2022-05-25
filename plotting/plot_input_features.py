from tempfile import TemporaryFile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import pandas as pd
import numpy as np

import argparse
import os

import json
from collections import OrderedDict

from tqdm import tqdm
import common

colour_schemes = {
  4: ['#a6cee3','#1f78b4','#b2df8a','#33a02c'],
  5: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99'],
  6: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c'],
  7: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f'],
  8: ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00']
}

def createDefaultConfig(data, bkg, sig):
  config = OrderedDict()
  columns_to_plot = filter(lambda x: ("weight" not in x), data.columns)
  for column in columns_to_plot:
    d = data[column][data[column]!=common.dummy_val]
    b = bkg[column][bkg[column]!=common.dummy_val]
    s = sig[column][sig[column]!=common.dummy_val]

    low = min([d.quantile(0.05), b.quantile(0.05), s.quantile(0.05)])
    high = max([d.quantile(0.95), b.quantile(0.95), s.quantile(0.95)])
    config[column] = {"range": [float(low),float(high)]}
  return config

def writeDefaultConfig(data, bkg, sig):
  config = createDefaultConfig(data, bkg, sig)
  with open("config.json", "w") as f:
    json.dump(config, f, indent=4)

def getConfig(data, bkg, sig, args):
  if args.config != None:
    with open(args.config, "r") as f:
      cfg = json.load(f)
  else:
    cfg = createDefaultConfig(data, bkg, sig)
    
  return cfg

def createBkgStack(bkg, column, proc_dict, group=True):
  bkg_stack = []
  bkg_stack_w = []
  bkg_stack_labels = []

  if not group:
    for proc in common.bkg_procs["all"]:
      bkg_stack.append(bkg[bkg.process_id==proc_dict[proc]][column])
      bkg_stack_w.append(bkg[bkg.process_id==proc_dict[proc]]["weight"])
      bkg_stack_labels.append(proc)
  else:
    for bkg_group in common.bkg_procs.keys():
      if bkg_group == "all": continue
      proc_ids = [proc_dict[proc] for proc in common.bkg_procs[bkg_group] if proc in proc_dict.keys()]
      bkg_stack.append(bkg[bkg.process_id.isin(proc_ids)][column])
      bkg_stack_w.append(bkg[bkg.process_id.isin(proc_ids)]["weight"])
      bkg_stack_labels.append(bkg_group)

  is_sorted = False
  while not is_sorted:
    is_sorted = True
    for i in range(len(bkg_stack)-1):
      if bkg_stack_w[i].sum() > bkg_stack_w[i+1].sum():
        is_sorted = False
        bkg_stack[i], bkg_stack[i+1] = bkg_stack[i+1], bkg_stack[i]
        bkg_stack_w[i], bkg_stack_w[i+1] = bkg_stack_w[i+1], bkg_stack_w[i]
        bkg_stack_labels[i], bkg_stack_labels[i+1] = bkg_stack_labels[i+1], bkg_stack_labels[i]

  return bkg_stack, bkg_stack_w, bkg_stack_labels

def getBkgError(bkg_stack, bkg_stack_w, edges):
  """
  Calculate error in each bin for each process
  Error = sqrt(N in bin) * avgw in bin
        = sqrt(N in bin) * (sumw in bin)/(N in bin)
        = (sumw in bin)/sqrt(N in bin)
  Total error = Sqrt( Sum(error^2) ) = Sqrt( Sum( (sumw in bin)^2/(N in bin) ) )
  """
  
  sumws = []
  errors = []  
  for i, bkg in enumerate(bkg_stack):
    N, edges = np.histogram(bkg, bins=edges)
    sumw, edges = np.histogram(bkg, bins=edges, weights=bkg_stack_w[i])
    
    sumws.append(sumw)
    errors.append(np.nan_to_num(sumw / np.sqrt(N)))

  sumw = np.array(sumws)
  errors = np.array(errors)
  
  sumw = np.sum(sumw, axis=0)
  error = np.sqrt(np.sum(errors**2, axis=0))
  return sumw, error  

def decayToMath(channel):
  if channel == "gg":
    return r"\gamma\gamma"
  else:
    return r"\tau\tau"

def getSigLabel(sig_proc):
  if "NMSSM" in sig_proc:
    split_name = sig_proc.split("_")
    Y_decay = decayToMath(split_name[3])
    H_decay = decayToMath(split_name[5])
    X_mass = int(split_name[7])
    Y_mass = int(split_name[9])
    label = r"$X_{%d} \rightarrow Y_{%d}(\rightarrow %s)  H(\rightarrow %s)$"%(X_mass, Y_mass, Y_decay, H_decay)
  elif "radion" in sig_proc:
    X_mass = int(sig_proc.split("M")[1].split("_")[0])
    label = r"$X_{%d} \rightarrow HH \rightarrow \gamma\gamma\tau\tau$"%X_mass
  else:
    label = sig_proc
  return label

def adjustLimits(x, ys, ax):
  data_to_display = ax.transData.transform
  display_to_data = ax.transData.inverted().transform

  tx = lambda x: data_to_display((x,0))[0]
  tx_inv = lambda x: display_to_data((x,0))[0]
  ty = lambda x: data_to_display((0,x))[1]
  ty_inv = lambda x: display_to_data((0,x))[1]

  xlow, xhigh = tx(ax.get_xlim()[0]), tx(ax.get_xlim()[1])
  ylow, yhigh = ty(ax.get_ylim()[0]), ty(ax.get_ylim()[1])
  
  #top side
  ybound = ylow + (yhigh-ylow)*0.60
  max_y = np.array(ys).max()
  top_distance_to_move = ty(max_y) - ybound
  
  #right side
  xbound = xlow + (xhigh-xlow)*0.75
  ybound = ylow + (yhigh-ylow)*0.20
  max_y = np.array(ys).T[x>tx_inv(xbound)].max()
  right_distance_to_move = ty(max_y) - ybound

  if right_distance_to_move <= 0:
    ax.legend(ncol=1, loc="upper right", markerfirst=False)
  elif right_distance_to_move < top_distance_to_move:
    ax.legend(ncol=1, loc="upper right", markerfirst=False)
    ax.set_ylim(top = ty_inv(yhigh + right_distance_to_move))
  else:
    ax.legend(ncol=3, loc="upper right", markerfirst=False)
    ax.set_ylim(top = ty_inv(yhigh + top_distance_to_move))

def plot_feature(data, bkg, sig, proc_dict, sig_procs, column, nbins, feature_range, save_path, auto_legend=True):
  if type(sig_procs) != list: sig_procs = [sig_procs]

  plt.rcParams["figure.figsize"] = (12.5,10)
  
  f, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
  
  data_hist, edges = np.histogram(data[column], bins=nbins, range=feature_range, weights=data.weight)
  bin_centres = (edges[:-1]+edges[1:])/2

  bkg_stack, bkg_stack_w, bkg_stack_labels = createBkgStack(bkg, column, proc_dict)

  bkg_stack_ungroup, bkg_stack_w_ungroup, bkg_stack_labels_ungroup = createBkgStack(bkg, column, proc_dict, group=False)
  bkg_sumw, bkg_error = getBkgError(bkg_stack_ungroup, bkg_stack_w_ungroup, edges)
  #bkg_sumw, bkg_error = getBkgError(bkg_stack, bkg_stack_w, edges)

  ratio = data_hist / bkg_sumw
  ratio_err = np.sqrt(data_hist) / bkg_sumw

  axs[0].fill_between(edges, np.append(bkg_sumw-bkg_error, 0), np.append(bkg_sumw+bkg_error, 0), step="post", alpha=0.5, color="grey", zorder=8) #background uncertainty
  axs[0].hist(bkg_stack, edges, weights=bkg_stack_w, label=bkg_stack_labels, stacked=True, color=colour_schemes[len(bkg_stack)], zorder=7) #background
  axs[0].errorbar(bin_centres, data_hist, np.sqrt(data_hist), label="Data", fmt='ko', zorder=10) #data
  axs[0].set_ylabel("Events")

  axs[1].errorbar(bin_centres, ratio, ratio_err, label="Data", fmt='ko')
  axs[1].fill_between(edges, np.append(1-bkg_error/bkg_sumw, 1), np.append(1+bkg_error/bkg_sumw, 1), step="post", alpha=0.5, color="grey")

  axs[1].set_xlabel(column)
  axs[1].set_ylabel("Data / MC")

  plt.sca(axs[0])
  mplhep.cms.label(llabel="Work in Progress", data=True, lumi=59, loc=0)

  for sig_proc in sig_procs:
    try: _ = [b.remove() for b in bars]
    except: pass
    sig_hist, edges = np.histogram(sig[sig.process_id==proc_dict[sig_proc]][column], bins=nbins, range=feature_range, weights=sig[sig.process_id==proc_dict[sig_proc]]["weight"])
    sig_sf = data_hist.max() / sig_hist.max()
    counts, bins, bars = axs[0].hist(edges[:-1], edges, weights=sig_hist*sig_sf, label=getSigLabel(sig_proc), histtype='step', color='r', lw=3, zorder=9) #signal

    if not auto_legend: axs[0].legend()
    
    axs[0].set_yscale("linear")
    axs[0].relim()
    axs[0].autoscale()
    axs[0].get_ylim()
    if auto_legend: adjustLimits(bin_centres, [sig_hist*sig_sf, data_hist], axs[0])
    plt.savefig("%s_%s.png"%(save_path, sig_proc))
    #plt.savefig("%s.pdf"%save_path)

    axs[0].set_yscale("log")
    axs[0].relim()
    axs[0].autoscale()
    axs[0].get_ylim()
    if auto_legend: adjustLimits(bin_centres, [sig_hist*sig_sf, data_hist], axs[0])
    plt.savefig("%s_%s_log.png"%(save_path, sig_proc))
    #plt.savefig("%s_log.pdf"%save_path)

  plt.close()

def plot(data, bkg, sig, proc_dict, args):
  cfg = getConfig(data, bkg, sig, args)

  for column in tqdm(cfg.keys()):
  #for column in ["Diphoton_phi", "Diphoton_pt_mgg"]:
  #for column in ["LeadPhoton_ditau_dR","SubleadPhoton_ditau_dR","LeadPhoton_tau_candidate_1_dR","LeadPhoton_tau_candidate_1_dR","SubleadPhoton_tau_candidate_1_dR","SubleadPhoton_tau_candidate_2_dR"]:
    nbins = 50
    feature_range = cfg[column]["range"]
    save_path = "%s/%s"%(args.output, column)
    plot_feature(data, bkg, sig, proc_dict, args.sig_procs, column, nbins, feature_range, save_path)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', type=str)
  parser.add_argument('--summary', '-s', type=str)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+")
  parser.add_argument('--output', '-o', type=str, default="plots")
  parser.add_argument('--config', '-c', type=str)
  parser.add_argument('--norm', default=False, action="store_true")
  parser.add_argument('--weight', default="weight_central", type=str)
  args = parser.parse_args()

  with open(args.summary, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  #sometimes bkg processes won't appear in parquet file because none passed selection
  for bkg_proc in common.bkg_procs["all"]:
    if bkg_proc not in proc_dict.keys():
      proc_dict[bkg_proc] = -9999
  
  print(">> Loading dataframes")  
  columns = common.all_columns_no_weight + [args.weight]
  columns_to_exclude = ["event", "MX", "MY"]
  columns = list(set(columns).difference(columns_to_exclude))
  df = pd.read_parquet(args.input, columns=columns)
  df.rename({args.weight: "weight"}, axis=1, inplace=True)

  print(">> Splitting into data, background and signal")
  data = df[df.process_id==proc_dict["Data"]]
  bkg_proc_ids = [proc_dict[bkg_proc] for bkg_proc in common.bkg_procs["all"]]
  bkg = df[df.process_id.isin(bkg_proc_ids)]
  sig_proc_ids = [proc_dict[proc] for proc in args.sig_procs]
  sig = df[df.process_id.isin(sig_proc_ids)]
  
  del df

  #blinding
  #data = data[(data.Diphoton_mass < 120) | (data.Diphoton_mass>130)]
  #bkg = bkg[(bkg.Diphoton_mass < 120) | (bkg.Diphoton_mass>130)]

  print("Data sumw: %f"%data.weight.sum())
  print("Bkg MC sumw: %f"%bkg.weight.sum())
  #normalise bkg mc to data
  if args.norm:
    bkg.loc[:, "weight"] = bkg.weight * (data.weight.sum() / bkg.weight.sum())

  os.makedirs(args.output, exist_ok=True)
   
  np.seterr(all='ignore')
  plot(data, bkg, sig, proc_dict, args)
  
  