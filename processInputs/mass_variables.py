"""
Number of mass variables I have experimented with. 
The variables use some combination of the visible particles and the MET.
"""

import numpy as np
import common

# def add_reco_MX_SVFit(df):
  # gg_px = df.Diphoton_pt * np.cos(0)
  # gg_py = df.Diphoton_pt * np.sin(0)
  # gg_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
  # gg_E = np.sqrt(df.Diphoton_mass**2 + gg_px**2 + gg_py**2 + gg_pz**2)

  # tt_px = df.pt_tautau_SVFit * np.cos(df.dPhi_ggtautau_SVFit)
  # tt_py = df.pt_tautau_SVFit * np.sin(df.dPhi_ggtautau_SVFit)
  # tt_eta = df.eta_tautau_SVFit_bdt * np.sign(df.Diphoton_eta)
  # tt_pz = df.pt_tautau_SVFit * np.sinh(tt_eta)
  # tt_E = np.sqrt(df.m_tautau_SVFit + tt_px**2 + tt_py**2 + tt_pz**2)

  # X_px = gg_px + tt_px
  # X_py = gg_py + tt_py
  # X_pz = gg_pz + tt_pz
  # X_E = gg_E + tt_E

  # df["reco_MX_SVFit"] = np.sqrt(X_E**2 - X_px**2 - X_py**2 - X_pz**2)
  # df["reco_MX_SVFit_mgg"] = df["reco_MX_SVFit"] / df.Diphoton_mass
  # df.loc[df.category == 8, "reco_MX_SVFit"] = common.dummy_val
  # df.loc[df.category == 8, "reco_MX_SVFit_mgg"] = common.dummy_val

def add_reco_MX(df):
  gg_px = df.Diphoton_pt * np.cos(0)
  gg_py = df.Diphoton_pt * np.sin(0)
  gg_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
  gg_E = np.sqrt(df.Diphoton_mass**2 + gg_px**2 + gg_py**2 + gg_pz**2)

  tt_px = df.ditau_pt * np.cos(df.Diphoton_ditau_dphi)
  tt_py = df.ditau_pt * np.sin(df.Diphoton_ditau_dphi)
  tt_pz = df.ditau_pt * np.sinh(df.ditau_eta)
  tt_E = np.sqrt(df.ditau_mass + tt_px**2 + tt_py**2 + tt_pz**2)

  X_px = gg_px + tt_px
  X_py = gg_py + tt_py
  X_pz = gg_pz + tt_pz
  X_E = gg_E + tt_E

  df["reco_MX"] = np.sqrt(X_E**2 - X_px**2 - X_py**2 - X_pz**2)
  df["reco_MX_mgg"] = df["reco_MX"] / df.Diphoton_mass
  df.loc[df.category == 8, "reco_MX"] = common.dummy_val
  df.loc[df.category == 8, "reco_MX_mgg"] = common.dummy_val
  df.loc[df.ditau_mass == common.dummy_val, "reco_MX"] = common.dummy_val
  df.loc[df.ditau_mass == common.dummy_val, "reco_MX_mgg"] = common.dummy_val

def add_reco_MX_MET(df):
  H1_px = df.Diphoton_pt * np.cos(df.Diphoton_phi)
  H1_py = df.Diphoton_pt * np.sin(df.Diphoton_phi)
  H1_pz = df.Diphoton_pt * np.sinh(df.Diphoton_eta)
  H1_P = df.Diphoton_pt * np.cosh(df.Diphoton_eta)
  H1_E = np.sqrt(H1_P**2 + df.Diphoton_mass**2)

  H2_px = df.ditau_pt * np.cos(df.ditau_phi) + df.MET_pt * np.cos(df.MET_phi)
  H2_py = df.ditau_pt * np.sin(df.ditau_phi) + df.MET_pt * np.sin(df.MET_phi)
  H2_pz = df.ditau_pt * np.sinh(df.ditau_eta) * 2
  H2_P = np.sqrt(H2_px**2 + H2_py**2 + H2_pz**2)
  H2_E = np.sqrt(H2_P**2 + df.ditau_mass**2)

  HH_px = H1_px + H2_px
  HH_py = H1_py + H2_py
  HH_pz = H1_pz + H2_pz
  HH_E = H1_E + H2_E

  df["reco_MX_MET"] = np.sqrt(HH_E**2 - HH_px**2 - HH_py**2 - HH_pz**2)
  df["reco_MX_MET_mgg"] = df["reco_MX_MET"] / df["Diphoton_mass"]
  df.loc[df.category == 8, "reco_MX_MET"] = common.dummy_val
  df.loc[df.category == 8, "reco_MX_MET_mgg"] = common.dummy_val