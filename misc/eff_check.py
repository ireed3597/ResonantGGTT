import pandas as pd
import sys
import json
import numpy as np

sig_procs_1 = ["radionM300_HHggTauTau", "radionM400_HHggTauTau", "radionM500_HHggTauTau", "radionM800_HHggTauTau", "radionM1000_HHggTauTau"]
sig_procs_2 = ['NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_100']
sig_procs = sig_procs_1 + sig_procs_2

bkg_procs = ['Diphoton_MGG-40to80', 'Diphoton_MGG-80toInf', 'GJets_HT-40To100', 'GJets_HT-100To200', 'GJets_HT-200To400', 'GJets_HT-400To600', 'GJets_HT-600ToInf', 'TTGG', 'TTGamma', 'TTJets', 'VBFH_M125', 'VH_M125', 'WGamma', 'ZGamma', 'ggH_M125', 'ttH_M125']
bkg_procs += ["QCD_Pt-30To40_MGG-80toInf", "QCD_Pt-30ToInf_MGG-40to80", "QCD_Pt-40ToInf_MGG-80toInf"]

df = pd.read_parquet(sys.argv[1])

with open(sys.argv[2]) as f:
  proc_dict = json.load(f)['sample_id_map']

for proc in proc_dict.keys():
  print(proc, df[df.process_id==proc_dict[proc]].weight_central.sum()/(41.5*1000))

procs_to_keep = bkg_procs + ["Data"]
ids_to_keep = [proc_dict[proc] for proc in procs_to_keep if proc in proc_dict.keys()]

ids_thrown_away = set(np.unique(df.process_id)).difference(ids_to_keep)

print("Throwing away:")
for proc in proc_dict.keys():
  if proc_dict[proc] in ids_thrown_away:
    print(proc)

df = df[df.process_id.isin(ids_to_keep)]
print("Data sumw: %f"%df[df.process_id==proc_dict["Data"]].weight_central.sum())
print("MC sumw: %f"%df[df.process_id!=proc_dict["Data"]].weight_central.sum())

