import pandas as pd
import argparse
import numpy as np
import json
import common
import mass_variables

dphi = lambda x, y: abs(x-y) - 2*(abs(x-y) - np.pi) * (abs(x-y) // np.pi)

def add_MX_MY(df, proc_dict):
  #get sig_procs in parquet file
  sig_procs = [sig_proc for sig_proc in common.sig_procs["all"] if sig_proc in proc_dict.keys()]

  df["MX"] = common.dummy_val
  df["MY"] = common.dummy_val
  
  for sig_proc in sig_procs:
    MX, MY = common.get_MX_MY(sig_proc)
    
    df.loc[df.process_id==proc_dict[sig_proc], "MX"] = MX
    df.loc[df.process_id==proc_dict[sig_proc], "MY"] = MY

# def divide_pt_by_mgg(df):
#   pt_columns = ["Diphoton_pt", "LeadPhoton_pt", "SubleadPhoton_pt"]
#   for column in pt_columns:
#     df.loc[:, column] /= df.Diphoton_mass
#     df.rename({column:column+"_mgg"}, axis=1, inplace=True)

def add_Deltas(df):
  df["Diphoton_deta"] = df.LeadPhoton_eta-df.SubleadPhoton_eta
  #df["Diphoton_dphi"] = dphi(df.LeadPhoton_phi, df.SubleadPhoton_phi)
  df["Diphoton_dR"] = np.sqrt( dphi(df.LeadPhoton_phi, df.SubleadPhoton_phi)**2 + (df.LeadPhoton_eta-df.SubleadPhoton_eta)**2 )

  df["ditau_deta"] = df.lead_lepton_eta - df.sublead_lepton_eta
  #df["ditau_dphi"] = dphi(df.lead_lepton_phi,df.sublead_lepton_phi)
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

# def add_helicity_angles(df):
#   import vector
#   Diphoton = vector.array({
#     "pt": df.Diphoton_pt,
#     "phi": df.Diphoton_phi,
#     "eta": df.Diphoton_eta,
#     "M": df.Diphoton_mass
#   })
#   Ditau = vector.array({
#     "pt": df.ditau_pt,
#     "phi": df.ditau_phi,
#     "eta": df.ditau_eta,
#     "M": df.ditau_mass
#   })
#   LeadPhoton = vector.array({
#     "pt": df.LeadPhoton_pt,
#     "phi": df.LeadPhoton_phi,
#     "eta": df.LeadPhoton_eta,
#     "M": df.LeadPhoton_mass
#   })
#   LeadTau = vector.array({
#     "pt": df.ditau_lead_lepton_pt,
#     "phi": df.ditau_lead_lepton_phi,
#     "eta": df.ditau_lead_lepton_eta,
#     "M": df.ditau_lead_lepton_mass
#   })

#   df["Diphoton_helicity_angle"] = np.cos(LeadPhoton.boost(-Diphoton).deltaangle(Diphoton))
#   df["ditau_helicity_angle"] = np.cos(LeadTau.boost(-Ditau).deltaangle(Ditau))
#   df["Diphoton_ditau_helicity_angle"] = np.cos(Ditau.boost(-(Diphoton+Ditau)).deltaangle(Diphoton+Ditau))

#   #Colin Soper angle
#   Dihiggs = Diphoton + Ditau
#   #boost such that dihiggs pz = 0
#   Dihiggs = Dihiggs.boostZ(beta=-Dihiggs.to_beta3().z)
#   Diphoton = Diphoton.boostZ(beta=-Dihiggs.to_beta3().z)
#   Ditau = Ditau.boostZ(beta=-Dihiggs.to_beta3().z)
#   #boost such that dihiggs at rest
#   Diphoton = Diphoton.boost(-Dihiggs)
#   Ditau = Ditau.boost(-Dihiggs)
#   df["Diphoton_ditau_Colin_Soper"] = np.cos(Ditau.deltaangle(Diphoton))

#   df.loc[df.category==8, "ditau_helicity_angle"] = common.dummy_val
#   df.loc[df.category==8, "Diphoton_ditau_helicity_angle"] = common.dummy_val
#   df.loc[df.category==8, "Diphoton_ditau_Colin_Soper"] = common.dummy_val

def add_MET_variables(df):
 # met_dphi variables already exist for diphoton and lead_lepton
 df["ditau_met_dPhi"] = dphi(df.MET_phi, df.ditau_phi)
 df["sublead_lepton_met_dPhi"] = dphi(df.MET_phi, df.sublead_lepton_phi)
 df.loc[df.category==8, "sublead_lepton_met_dPhi"] = common.dummy_val

def applyPixelVeto(df):
  pixel_veto = (df.LeadPhoton_pixelSeed==0) & (df.SubleadPhoton_pixelSeed==0)
  df.drop(df[~pixel_veto].index, inplace=True)

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

def main(args):
  if not args.test:
    df = pd.read_parquet(args.parquet_input)
  else:
    from pyarrow.parquet import ParquetFile
    import pyarrow as pa
    pf = ParquetFile(args.parquet_input) 
    iter = pf.iter_batches(batch_size = 10)
    first_ten_rows = next(iter) 
    df = pa.Table.from_batches([first_ten_rows]).to_pandas() 

  original_columns = list(df.columns)

  checkNans(df)
  checkInfs(df)

  with open(args.summary_input, "r") as f:
    proc_dict = json.load(f)["sample_id_map"]

  add_MX_MY(df, proc_dict)

  applyPixelVeto(df)

  add_ditau_phi(df)
  add_MET_variables(df)
  add_Deltas(df)
  mass_variables.add_reco_MX(df)  
  mass_variables.add_reco_MX_met4(df)
  mass_variables.add_Mggt(df)
  mass_variables.add_Mggt_met1(df)
  #add_helicity_angles(df)
  #divide_pt_by_mgg(df)
  merge2016(df)

  fixDtypes(df)
  reduceMemory(df)

  print("Additional columns:")
  print(set(df.columns).difference(original_columns))

  df.to_parquet(args.parquet_output)
  return df

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--parquet-input', '-i', type=str, required=True)
  parser.add_argument('--parquet-output', '-o', type=str, required=True)
  parser.add_argument('--summary-input', '-s', type=str, required=True)

  parser.add_argument('--test', action="store_true", default=False)

  args = parser.parse_args()

  df = main(args)