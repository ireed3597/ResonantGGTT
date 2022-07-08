import json
import pandas as pd
import sys
import common

df = pd.read_parquet(sys.argv[1])

print(df[df.process_id==33].weight.sum()/140)

boundaries = [0.0, 0.9962, 0.998, 1.0][::-1]
with open(sys.argv[2], "r") as f:
  proc_dict = json.load(f)["sample_id_map"]

sm_higgs_ids = [proc_dict[proc] for proc in common.bkg_procs["SM Higgs"]]

df = df[df.process_id.isin(sm_higgs_ids)]

for score_name in df.columns:
  if "intermediate_transformed_score" in score_name:
    for i in range(len(boundaries)-1):
      selection = (df[score_name] <= boundaries[i]) & (df[score_name] > boundaries[i+1])
      df.loc[selection, "SR"] = i

    print(score_name)
    for SR in df.SR.unique():
      print(SR, df.loc[df.SR==SR, "weight"].sum() / 140)