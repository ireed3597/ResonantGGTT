import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import numpy as np

df = pd.read_parquet(sys.argv[1])
df = df[df.process_id==18]
for i in range(4):
  if i==0:
    sf = 0.02
  elif i==2:
    sf = 3
  elif i==3:
    sf = 1
  else:
    continue

  r = (i*0.25, (i+1)*0.25)
  cut = (df.score_XToHHggTauTau_M600>r[0]) & (df.score_XToHHggTauTau_M600<r[1])
  print(cut.sum())
  hist, bin_edges = np.histogram(df.Diphoton_mass[cut], bins=25, range=(110,150), weights=df[cut].weight)
  c = (bin_edges[:-1]+bin_edges[1:])/2

  plt.hist(bin_edges[:-1], bin_edges, histtype='step', weights=hist*sf, label="%.2f < MVA < %.2f"%r)
  plt.errorbar(c, hist*sf, (hist/np.sqrt(cut.sum()))*sf, fmt='.')
plt.xlabel("mgg")
plt.legend()
plt.savefig("mgg_sculpting.png")