import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np

import sys
import common
import json

df = pd.read_parquet(sys.argv[1], columns=["process_id", "weight_central"])
with open(sys.argv[2], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

sig_ids = [proc_dict[proc] for proc in common.sig_procs["Graviton"]]
mx = [common.get_MX_MY(proc)[0] for proc in common.sig_procs["Graviton"]]

lumi = sum([common.lumi_table[year] for year in common.lumi_table.keys()])
eff = [df[df.process_id==i].weight_central.sum()/lumi for i in sig_ids]

N = [sum(df.process_id==i) for i in sig_ids]
print(N)
eff_err = eff / np.sqrt(N)

#plt.scatter(mx, eff, marker='.')
plt.errorbar(mx, eff, eff_err, fmt='o')
plt.xlabel(r"$m_X$")
plt.ylabel("Signal efficiency")
plt.savefig("sig_eff.png")
plt.savefig("sig_eff.pdf")
plt.clf()

mx=np.array(mx)
eff=np.array(eff)
eff_err=np.array(eff_err)
plt.errorbar(mx[mx<450], eff[mx<450], eff_err[mx<450], fmt='.')
plt.xlabel("MX")
plt.ylabel("Signal efficiency")
plt.savefig("sig_eff_zoom.png")
plt.clf()

plt.hist(df[df.process_id==proc_dict['XToHHggTauTau_M260']].weight_central, bins=100)
plt.savefig("weight_hist.png")