"""
Check correlation between the mass variables I have created and diphoton mass
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import pandas as pd

df = pd.read_parquet(sys.argv[1])
#df = df[df.category==8]

df["reco_Mggtau_mgg"] = df.reco_Mggtau - df.Diphoton_mass

plt.hist(df[df.process_id==29].reco_Mggtau_mgg, bins=25, range=(0, 100), density=True, alpha=0.5, label="signal")
plt.hist(df[df.process_id==0].reco_Mggtau_mgg, bins=25, range=(0, 100), density=True, alpha=0.5)
plt.legend()
plt.savefig("mass_corr_test2.png")
plt.clf()

df = df[df.process_id==0]

reco_masses = list(filter(lambda x: "reco" in x, df.columns))

methods = ['pearson', 'kendall', 'spearman']
df = df[reco_masses+["Diphoton_mass", "Diphoton_pt_mgg", "MET_pt"]]

for m in methods:
  print(m)
  print(df.corr(m)[["Diphoton_mass", "Diphoton_pt_mgg", "MET_pt"]])

plt.hist(df.Diphoton_mass, bins=25, range=(100,180), density=True, alpha=0.5, label="Nominal")
cut = (df.reco_Mggtau_mgg > 30) & (df.reco_Mggtau_mgg < 50)
#cut = (df.reco_MggtauMET > 200) & ( df.reco_MggtauMET < 250)
#cut = df.MET_pt > 80
#cut = df.reco_MX > 300
plt.hist(df.Diphoton_mass[cut], bins=25, range=(100,180), density=True, alpha=0.5, label="Cut")
plt.legend()
plt.xlabel("mgg")
plt.savefig("mass_corr.png")