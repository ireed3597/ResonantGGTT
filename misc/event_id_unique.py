import pandas as pd
import numpy as np
import sys

import json
with open(sys.argv[3], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

proc_ids = [proc_dict[proc] for proc in proc_dict.keys() if "XToHHggTauTau" in proc]
proc_ids.remove(28)

df1 = pd.read_parquet(sys.argv[1], columns=["Diphoton_eta", "LeadPhoton_pt", "event", "year", "process_id", "weight_central"])
df1 = df1[df1.process_id.isin(proc_ids)]

#df1 = pd.read_parquet(sys.argv[1], columns=["event", "year", "process_id"])
#df1 = df1[df1.process_id==46]
#df1 = df1[df1.year==b"2018"]
#df1.sort_values(["year","process_id","event"], inplace=True)
#df1 = df1.iloc[0:len(df1)//10]
df1_multi = df1.set_index(["year", "process_id", "event"])

u, c = np.unique(df1.event, return_counts=True)

# print(c)
# print(len(c))
# print(sum(c>2))

df2 = pd.read_parquet(sys.argv[2], columns=["Diphoton_eta", "event", "year", "process_id", "weight_central"])
df2 = df2[df2.process_id.isin(proc_ids)]

#df2 = pd.read_parquet(sys.argv[2], columns=["event", "year", "process_id"])
#df2 = df2[df2.process_id==46]
#df2 = df2[df2.year==b"2018"]
#df2.sort_values(["year","process_id","event"], inplace=True)
#df2 = df2.iloc[0:len(df2)//10]
df2_multi = df2.set_index(["year", "process_id", "event"])

print(df2.process_id.unique())

#print(len(df1))
#print(len(set(df1.event).intersection(df2.event)))

#print(sum(abs(df1.Diphoton_eta-df2.Diphoton_eta)>0.0001))

print(df1)
print(df1[df1.process_id==46].weight_central.sum())
print(df2[df2.process_id==46].weight_central.sum())
print(df1.weight_central.sum())
print(df2.weight_central.sum())


df = df1
df_int = df2

df_int.set_index(["year", "process_id", "event"], inplace=True)
df_temp = pd.DataFrame({"ind":df.index, "year":df.year, "process_id":df.process_id, "event":df.event})
df_temp.set_index(["year", "process_id", "event"], inplace=True)
ind = df_temp.loc[df_int.index.intersection(df_temp.index), "ind"]

df = df.loc[ind]

print(df[df.process_id==46].weight_central.sum())
print(df.weight_central.sum())