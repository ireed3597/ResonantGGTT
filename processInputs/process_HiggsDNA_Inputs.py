import pandas as pd
import argparse
import numpy as np
import json
import common
import mass_variables
import sys

dphi = lambda x, y: abs(x-y) - 2*(abs(x-y) - np.pi) * (abs(x-y) // np.pi)

def add_Deltas(df):
  df["Diphoton_deta"] = df.LeadPhoton_eta-df.SubleadPhoton_eta
  df["Diphoton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi, df.SubleadPhoton_phi)**2 + (df.LeadPhoton_eta-df.SubleadPhoton_eta)**2 )

  df["ditau_deta"] = df.lead_lepton_eta - df.sublead_lepton_eta
  df.loc[df.category==8, "ditau_deta"] = common.dummy_val
  df.loc[df.category==8, "ditau_dphi"] = common.dummy_val

  df["Diphoton_lead_lepton_deta"] = df.Diphoton_eta-df.lead_lepton_eta
  df["Diphoton_lead_lepton_dphi"] = dphi(df.Diphoton_phi,df.lead_lepton_phi)
  df["Diphoton_lead_lepton_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.lead_lepton_phi)**2 + (df.Diphoton_eta-df.lead_lepton_eta)**2 )

  df["Diphoton_sublead_lepton_deta"] = df.Diphoton_eta-df.sublead_lepton_eta
  df["Diphoton_sublead_lepton_dphi"] = dphi(df.Diphoton_phi,df.sublead_lepton_phi)
  df["Diphoton_sublead_lepton_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.sublead_lepton_phi)**2 + (df.Diphoton_eta-df.sublead_lepton_eta)**2 )
  df.loc[df.category==8, "Diphoton_sublead_lepton_deta"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_sublead_lepton_dphi"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_sublead_lepton_dR"] = common.dummy_val

  df["Diphoton_ditau_deta"] = df.Diphoton_eta-df.ditau_eta
  df["Diphoton_ditau_dphi"] = dphi(df.Diphoton_phi,df.ditau_phi)
  df["Diphoton_ditau_dR"] = np.sqrt( dphi(df.Diphoton_phi,df.ditau_phi)**2 + (df.Diphoton_eta-df.ditau_eta)**2 )
  df.loc[df.category==8, "Diphoton_ditau_deta"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_ditau_dphi"] = common.dummy_val
  df.loc[df.category==8, "Diphoton_ditau_dR"] = common.dummy_val

  #zgamma variables
  df["LeadPhoton_ditau_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.ditau_phi)**2 + (df.LeadPhoton_eta-df.ditau_eta)**2 )
  df["SubleadPhoton_ditau_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.ditau_phi)**2 + (df.SubleadPhoton_eta-df.ditau_eta)**2 )
  df["LeadPhoton_lead_lepton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.lead_lepton_phi)**2 + (df.LeadPhoton_eta-df.lead_lepton_eta)**2 )
  df["SubleadPhoton_lead_lepton_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.lead_lepton_phi)**2 + (df.SubleadPhoton_eta-df.lead_lepton_eta)**2 )
  df["LeadPhoton_sublead_lepton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi,df.sublead_lepton_phi)**2 + (df.LeadPhoton_eta-df.sublead_lepton_eta)**2 )
  df["SubleadPhoton_sublead_lepton_dR"] = np.sqrt( dphi(df.SubleadPhoton_phi,df.sublead_lepton_phi)**2 + (df.SubleadPhoton_eta-df.sublead_lepton_eta)**2 )
  df.loc[df.category==8, "LeadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.category==8, "SubleadPhoton_ditau_dR"] = common.dummy_val
  df.loc[df.category==8, "LeadPhoton_sublead_lepton_dR"] = common.dummy_val
  df.loc[df.category==8, "SubleadPhoton_sublead_lepton_dR"] = common.dummy_val

def add_MET_variables(df):
  # met_dphi variables already exist for diphoton and lead_lepton
  df["ditau_met_dPhi"] = dphi(df.MET_phi, df.ditau_phi)
  df.loc[df.category==8, "ditau_met_dPhi"] = common.dummy_val

  df["sublead_lepton_met_dPhi"] = dphi(df.MET_phi, df.sublead_lepton_phi)
  df.loc[df.category==8, "sublead_lepton_met_dPhi"] = common.dummy_val

def applyPixelVeto(df):
  pixel_veto = (df.LeadPhoton_pixelSeed==0) & (df.SubleadPhoton_pixelSeed==0)
  #df.drop(df[~pixel_veto].index, inplace=True)
  return df[pixel_veto]

def apply90WPID(df):
  selection = (df.Diphoton_max_mvaID > -0.26) & (df.Diphoton_min_mvaID > -0.26)
  #df.drop(df[~selection].index, inplace=True)
  return df[selection]

def reduceMemory(df):
  print(df.info())
  for column in df.columns:
    if df[column].dtype == "float64":
      print("%s float64 -> float32"%column.ljust(50))
      df.loc[:, column] = df[column].astype("float32")
    elif df[column].dtype == "int64":
      print("%s  int64 -> uint8"%column.ljust(50))
      df.loc[:, column] = df[column].astype("uint8")
    else:
      print("%s %s -> %s"%(column.ljust(50), df[column].dtype, df[column].dtype))
  print(df.info())

def fixDtypes(df):
  df.loc[:, "lead_lepton_id"] = df["lead_lepton_id"].astype("int8")
  df.loc[:, "sublead_lepton_id"] = df["sublead_lepton_id"].astype("int8")
  df.loc[:, "lead_lepton_charge"] = df["lead_lepton_charge"].astype("int8")
  df.loc[:, "sublead_lepton_charge"] = df["sublead_lepton_charge"].astype("int8")
  
  df.loc[:, "process_id"] = df["process_id"].astype("uint8")
  df.loc[:, "year"] = df["year"].astype("uint16")

  df.loc[:, "LeadPhoton_pixelSeed"] = df["LeadPhoton_pixelSeed"].astype("uint8")
  df.loc[:, "SubleadPhoton_pixelSeed"] = df["SubleadPhoton_pixelSeed"].astype("uint8")

def checkNans(df):
  for column in df.columns:
    try:
      if np.isnan(df[column]).any():
        print(df.loc[np.isnan(df[column]), column])
        df.loc[:, column].replace(np.nan, common.dummy_val, inplace=True)
    except:
      pass

def checkInfs(df):
  for column in df.columns:
    try:
      if np.isinf(df[column]).any():
        print(df.loc[np.isinf(df[column]), column])
        df.drop(df.loc[np.isinf(df[column])].index, inplace=True)    
    except:
      pass  

def merge2016(df):
  df.loc[df.year==b"2016UL_pre", "year"] = "2016"
  df.loc[df.year==b"2016UL_pos", "year"] = "2016"

def add_ditau_phi(df):
  tau1_px = df.lead_lepton_pt * np.cos(df.lead_lepton_phi)
  tau1_py = df.lead_lepton_pt * np.sin(df.lead_lepton_phi)
  tau2_px = df.sublead_lepton_pt * np.cos(df.sublead_lepton_phi)
  tau2_py = df.sublead_lepton_pt * np.sin(df.sublead_lepton_phi)

  ditau_px = tau1_px + tau2_px
  ditau_py = tau1_py + tau2_py
  df["ditau_phi"] = np.arctan2(ditau_py, ditau_px)
  df.loc[df.category==8, "ditau_phi"] = common.dummy_val

def dividePhotonPT(df):
  df["LeadPhoton_pt_mgg"] = df["LeadPhoton_pt"] / df["Diphoton_mass"]
  df["SubleadPhoton_pt_mgg"] = df["SubleadPhoton_pt"] / df["Diphoton_mass"]

def prefiringWeights(df):
  df.loc[:, "weight_L1_prefiring_sf_central"] = 1.0
  df.loc[:, "weight_L1_prefiring_sf_up"] = 1.0
  df.loc[:, "weight_L1_prefiring_sf_down"] = 1.0

def selectSigProcs(df, proc_dict, sig_procs):
  data_bkg_ids = [proc_dict[proc] for proc in common.bkg_procs["all"]+["Data"]]
  sig_proc_ids = [proc_dict[proc] for proc in sig_procs]
  return df[df.process_id.isin(data_bkg_ids+sig_proc_ids)]

def main(parquet_input, parquet_output, summary_input, do_test, keep_features, sig_procs=None):
  if not do_test:
    df = pd.read_parquet(parquet_input)
  else:
    from pyarrow.parquet import ParquetFile
    import pyarrow as pa
    pf = ParquetFile(parquet_input) 
    iter = pf.iter_batches(batch_size = 10)
    first_ten_rows = next(iter) 
    df = pa.Table.from_batches([first_ten_rows]).to_pandas() 

  original_columns = list(df.columns)

  with open(summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  if sig_procs != None:
    df = selectSigProcs(df, proc_dict, sig_procs)

  print(1)
  df = applyPixelVeto(df)
  df = apply90WPID(df)

  print(2)
  prefiringWeights(df)
  print(2.2)
  checkNans(df)
  print(2.3)
  checkInfs(df)

  common.add_MX_MY(df, proc_dict)

  print(3)
  add_ditau_phi(df)
  print(4)
  add_MET_variables(df)
  print(5)
  add_Deltas(df)
  print(6)
  dividePhotonPT(df)
  print(7)
  mass_variables.add_reco_MX(df)  
  mass_variables.add_reco_MX_met4(df)
  mass_variables.add_Mggt(df)
  mass_variables.add_Mggt_met1(df)
  #add_helicity_angles(df)
  #divide_pt_by_mgg(df)
  print(8)
  merge2016(df)
  print(9)

  print("Additional columns:")
  print(set(df.columns).difference(original_columns))

  print(10)
  fixDtypes(df)

  if keep_features != None:
    keep_features = common.train_features[keep_features]
    keep_features += list(filter(lambda x: "reco_" in x, df.columns)) #reco mass vars
    keep_features += list(filter(lambda x: "weight" in x, df.columns)) #add weights
    keep_features += ["Diphoton_mass", "MX", "MY", "event", "year", "category", "process_id"] #add other neccessary columns
    keep_features = list(set(keep_features)) #remove overlap in columns
    df = df[keep_features]

  print(11)
  reduceMemory(df)

  print("Final columns:")
  print(df.columns)

  print(12)

  df.to_parquet(parquet_output)
  return df

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--parquet-output', '-o', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)
  parser.add_argument('--test', action="store_true", default=False)
  parser.add_argument('--keep-features', '-f', type=str, default=None)
  parser.add_argument('--batch', action="store_true")
  parser.add_argument('--batch-slots', type=int, default=1)
  parser.add_argument('--sig-procs', '-p', type=str, nargs="+", default=None)

  args = parser.parse_args()

  if args.batch:
    common.submitToBatch([sys.argv[0]] + common.parserToList(args), extra_memory=args.batch_slots)
  else:
    main(args.parquet_input, args.parquet_output, args.summary_input, args.test, args.keep_features, args.sig_procs)