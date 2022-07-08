import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import numpy as np

vars = ["LeadPhoton_pt", "SubleadPhoton_pt", "Diphoton_pt", "Diphoton_mass"]
columns = ["weight_central", "process_id"] + vars
columns=None

cut = lambda df: df[(df.process_id==33)&(df.year==2018)].sort_values("event")

nominal = pd.read_parquet("Outputs/ParamNN_all_masses_systematics_smear_fix/merged_nominal.parquet", columns=columns)
nominal = cut(nominal)
up = pd.read_parquet("Outputs/ParamNN_all_masses_systematics_smear_fix/merged_smear_up.parquet", columns=columns)
up = cut(up)
down = pd.read_parquet("Outputs/ParamNN_all_masses_systematics_smear_fix/merged_smear_down.parquet", columns=columns)
down = cut(down)

events = set(nominal.event).intersection(up.event).intersection(down.event)
columns = set(nominal.columns).intersection(up.columns).intersection(down.columns)

event_match = lambda df: df[df.event.isin(events)][columns]
nominal = event_match(nominal)
up = event_match(up)
down = event_match(down)


print(nominal.head(5))
print(up.head(5))
print(down.head(5))