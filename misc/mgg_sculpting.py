import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np
import json
import common

from plotting.plot_input_features import *

def plot_2d(bkg, score_column, save_path):
  mgg = bkg.Diphoton_mass
  score = bkg[score_column]

  plt.hist2d(mgg,score, bins=50, weights=bkg.weight)
  plt.colorbar()
  plt.xlabel(r"$m_{\gamma\gamma}$")
  plt.ylabel(score_column)
  plt.savefig("%s.png"%save_path)

def plot_bkg(bkg, proc_dict, column, nbins, feature_range, save_path, auto_legend=True):
  plt.rcParams["figure.figsize"] = (12.5,10)
  
  bkg_stack, bkg_stack_w, bkg_stack_labels = createBkgStack(bkg, column, proc_dict)
  bkg_stack_ungroup, bkg_stack_w_ungroup, bkg_stack_labels_ungroup = createBkgStack(bkg, column, proc_dict, group=False)

  n, edges, patches = plt.hist(bkg_stack, nbins, range=feature_range, weights=bkg_stack_w, label=bkg_stack_labels, stacked=True, color=colour_schemes[len(bkg_stack)], zorder=7) #background
  bin_centres = (edges[:-1]+edges[1:])/2

  bkg_sumw, bkg_error = getBkgError(bkg_stack_ungroup, bkg_stack_w_ungroup, edges)

  plt.fill_between(edges, np.append(bkg_sumw-bkg_error, 0), np.append(bkg_sumw+bkg_error, 0), step="post", alpha=0.5, color="grey", zorder=8) #background uncertainty
  plt.ylabel("Events")

  mplhep.cms.label(llabel="Work in Progress", data=True, lumi=138, loc=0)

  plt.legend()
  plt.savefig("%s.png"%(save_path))
  #plt.savefig("%s_log.png"%(save_path))
  plt.close()

columns = common.getColumns(os.path.join(sys.argv[1],"merged_nominal.parquet"))
columns = list(filter(lambda x: not ("score" in x and ("intermediate" not in x)), columns))

df = pd.read_parquet(os.path.join(sys.argv[1],"merged_nominal.parquet"), columns=columns)
with open(sys.argv[2], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

# bkg_ids = [proc_dict[proc] for proc in common.bkg_procs['all']]
# df = df[df.process_id.isin(bkg_ids)]
# gjet_ids = [proc_dict[proc] for proc in common.bkg_procs['GJets']]
# df = df[~df.process_id.isin(gjet_ids)]
# smhiggs_ids = [proc_dict[proc] for proc in common.bkg_procs['SM Higgs']]
# df = df[~df.process_id.isin(smhiggs_ids)]
bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["Diphoton"]]
df = df[df.process_id.isin(bkg_ids)]

df = df[df.process_id==0]

os.makedirs(os.path.join(sys.argv[1], "mgg_sculpting"), exist_ok=True)

first_col = True
for column in df.columns:
  if "intermediate" in column:
    sig_proc = column.split("score_")[1]
    mx, my = common.get_MX_MY(sig_proc)
    if first_col:
      plot_bkg(df, proc_dict, "Diphoton_mass", 20, [my-40,my+40], os.path.join(sys.argv[1], "mgg_sculpting", "preselection"))
      first_col = False

    plot_2d(df, column, os.path.join(sys.argv[1], "mgg_sculpting", "%s_mgg_score_2d"%column))

    df_cut = df[df[column] > 0.99]
    plot_bkg(df_cut, proc_dict, "Diphoton_mass", 20, [my-40,my+40], os.path.join(sys.argv[1], "mgg_sculpting", "%s_high"%column))

    df_cut = df[df[column] < 0.1]
    plot_bkg(df_cut, proc_dict, "Diphoton_mass", 20, [my-40,my+40], os.path.join(sys.argv[1], "mgg_sculpting", "%s_low"%column))

# for i in range(5):
#   # if i==0:
#   #   sf = 1
#   # elif i==2:
#   #   sf = 1
#   # elif i==3:
#   #   sf = 1
#   # else:
#   #   continue
#   if i==4:
#     sf=1
#   else:
#     continue

#   r = (i*0.2, (i+1)*0.2)
#   cut = (df.score_XToHHggTauTau_M260>r[0]) & (df.score_XToHHggTauTau_M260<r[1])
#   print(cut.sum())
#   hist, bin_edges = np.histogram(df.Diphoton_mass[cut], bins=25, range=(110,150), weights=df[cut].weight, normed=True)
#   c = (bin_edges[:-1]+bin_edges[1:])/2

#   plt.hist(bin_edges[:-1], bin_edges, histtype='step', weights=hist*sf, label="%.2f < MVA < %.2f"%r)
#   plt.errorbar(c, hist*sf, (hist/np.sqrt(cut.sum()))*sf, fmt='.')
# plt.xlabel("mgg")
# plt.legend()
# plt.savefig("mgg_sculpting.png")