import os

dummy_val = -9.0 #value used for missing variables, e.g. sublead_lepton_pt when there is only one lepton

category_map = {
  1: "tau/mu",
  2: "tau/ele",
  3: "tau/tau",
  4: "mu/mu",
  5: "ele/ele",
  6: "mu/ele",
  7: "tau/iso_track",
  8: "tau"
}

sig_procs = {
  "Radion": [],
  "Graviton": ['XToHHggTauTau_M1000', 'XToHHggTauTau_M250', 'XToHHggTauTau_M260', 'XToHHggTauTau_M270', 'XToHHggTauTau_M280', 'XToHHggTauTau_M290', 'XToHHggTauTau_M300', 'XToHHggTauTau_M320', 'XToHHggTauTau_M350', 'XToHHggTauTau_M400', 'XToHHggTauTau_M450', 'XToHHggTauTau_M500', 'XToHHggTauTau_M550', 'XToHHggTauTau_M600', 'XToHHggTauTau_M650', 'XToHHggTauTau_M700', 'XToHHggTauTau_M750', 'XToHHggTauTau_M800', 'XToHHggTauTau_M900'],
  "NMSSM_all": []
  #"NMSSM_all": ['NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_100'],
  #"NMSSM_Y_gg": ['NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100'],
  #"NMSSM_Y_tautau": ['NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_50', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_100'],
}
sig_procs["all"] = sig_procs["Radion"] + sig_procs["Graviton"] + sig_procs["NMSSM_all"]

bkg_procs = {
  'Diphoton': ['DiPhoton'],
  'GJets': ['GJets_HT-100To200', 'GJets_HT-200To400', 'GJets_HT-400To600', 'GJets_HT-40To100', 'GJets_HT-600ToInf'],
  'TT': ['TTGG', 'TTGamma', 'TTJets'],
  'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
  'VGamma': ['WGamma', 'ZGamma']
}
bkg_procs["all"] = [proc for key in bkg_procs.keys() for proc in bkg_procs[key]]

all_columns = ['Diphoton_mass', 'Diphoton_pt', 'Diphoton_eta', 'Diphoton_phi', 'Diphoton_helicity', 'Diphoton_pt_mgg', 'Diphoton_max_mvaID', 'Diphoton_min_mvaID', 'Diphoton_dPhi', 'LeadPhoton_pt', 'LeadPhoton_eta', 'LeadPhoton_phi', 'LeadPhoton_mass', 'LeadPhoton_mvaID', 'LeadPhoton_pixelSeed', 'SubleadPhoton_pt', 'SubleadPhoton_eta', 'SubleadPhoton_phi', 'SubleadPhoton_mass', 'SubleadPhoton_mvaID', 'SubleadPhoton_pixelSeed', 'weight_central', 'n_electrons', 'n_muons', 'n_taus', 'n_iso_tracks', 'n_jets', 'n_bjets', 'MET_pt', 'MET_phi', 'diphoton_met_dPhi', 'lead_lepton_met_dphi', 'ditau_dphi', 'ditau_deta', 'ditau_dR', 'ditau_mass', 'ditau_pt', 'ditau_eta', 'lead_lepton_pt', 'lead_lepton_eta', 'lead_lepton_phi', 'lead_lepton_mass', 'lead_lepton_charge', 'lead_lepton_id', 'sublead_lepton_pt', 'sublead_lepton_eta', 'sublead_lepton_phi', 'sublead_lepton_mass', 'sublead_lepton_charge', 'sublead_lepton_id', 'category', 'jet_1_pt', 'jet_1_eta', 'jet_1_btagDeepFlavB', 'jet_2_pt', 'jet_2_eta', 'jet_2_btagDeepFlavB', 'b_jet_1_btagDeepFlavB', 'dilep_leadpho_mass', 'dilep_subleadpho_mass', 'event', 'process_id', 'year', 'weight_trigger_sf_down', 'weight_btag_deepjet_sf_SelectedJet_up_jes', 'weight_muon_id_sfSTAT_SelectedMuon_central', 'weight_central_initial', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_down', 'weight_btag_deepjet_sf_SelectedJet_up_hf', 'weight_muon_id_sfSYS_SelectedMuon_down', 'LeadPhoton_genPartFlav', 'weight_trigger_sf_central', 'weight_muon_id_sfSYS_SelectedMuon_up', 'weight_electron_veto_sf_Diphoton_Photon_central', 'weight_btag_deepjet_sf_SelectedJet_up_lfstats2', 'weight_tau_idDeepTauVSe_sf_AnalysisTau_down', 'weight_muon_iso_sfSYS_SelectedMuon_up', 'weight_L1_prefiring_sf_up', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_up', 'weight_muon_id_sfSYS_SelectedMuon_central', 'weight_trigger_sf_up', 'weight_muon_iso_sfSYS_SelectedMuon_down', 'weight_btag_deepjet_sf_SelectedJet_central', 'weight_electron_id_sf_SelectedElectron_down', 'weight_puWeight_central', 'weight_muon_iso_sfSTAT_SelectedMuon_up', 'weight_btag_deepjet_sf_SelectedJet_up_lf', 'weight_muon_iso_sfSTAT_SelectedMuon_central', 'weight_btag_deepjet_sf_SelectedJet_up_lfstats1', 'weight_btag_deepjet_sf_SelectedJet_down_lfstats2', 'weight_puWeight_down', 'weight_btag_deepjet_sf_SelectedJet_down_cferr2', 'weight_btag_deepjet_sf_SelectedJet_down_hfstats1', 'SubleadPhoton_genPartFlav', 'weight_btag_deepjet_sf_SelectedJet_up_hfstats2', 'weight_electron_veto_sf_Diphoton_Photon_up', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_down', 'weight_puWeight_up', 'weight_btag_deepjet_sf_SelectedJet_up_hfstats1', 'weight_muon_iso_sfSTAT_SelectedMuon_down', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_up', 'weight_electron_id_sf_SelectedElectron_central', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_central', 'weight_btag_deepjet_sf_SelectedJet_up_cferr2', 'weight_muon_iso_sfSYS_SelectedMuon_central', 'weight_btag_deepjet_sf_SelectedJet_down_jes', 'weight_btag_deepjet_sf_SelectedJet_down_hf', 'weight_central_no_lumi', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_central', 'weight_muon_id_sfSTAT_SelectedMuon_down', 'weight_L1_prefiring_sf_down', 'weight_muon_id_sfSTAT_SelectedMuon_up', 'weight_btag_deepjet_sf_SelectedJet_down_cferr1', 'weight_btag_deepjet_sf_SelectedJet_down_hfstats2', 'weight_btag_deepjet_sf_SelectedJet_down_lfstats1', 'weight_electron_veto_sf_Diphoton_Photon_down', 'weight_L1_prefiring_sf_central', 'weight_btag_deepjet_sf_SelectedJet_down_lf', 'weight_tau_idDeepTauVSe_sf_AnalysisTau_central', 'weight_electron_id_sf_SelectedElectron_up', 'weight_tau_idDeepTauVSe_sf_AnalysisTau_up', 'weight_btag_deepjet_sf_SelectedJet_up_cferr1', 'MX', 'MY', 'ditau_phi', 'ditau_met_dPhi', 'sublead_lepton_met_dPhi', 'Diphoton_deta', 'Diphoton_dR', 'Diphoton_lead_lepton_deta', 'Diphoton_lead_lepton_dphi', 'Diphoton_lead_lepton_dR', 'Diphoton_sublead_lepton_deta', 'Diphoton_sublead_lepton_dphi', 'Diphoton_sublead_lepton_dR', 'Diphoton_ditau_deta', 'Diphoton_ditau_dphi', 'Diphoton_ditau_dR', 'LeadPhoton_ditau_dR', 'SubleadPhoton_ditau_dR', 'LeadPhoton_lead_lepton_dR', 'SubleadPhoton_lead_lepton_dR', 'LeadPhoton_sublead_lepton_dR', 'SubleadPhoton_sublead_lepton_dR', 'reco_MX', 'reco_MX_MET', 'reco_Mggtau', 'reco_MggtauMET']
all_columns_no_weight = list(filter(lambda x: "weight" not in x, all_columns))

train_features = {
  'base': ['Diphoton_eta', 'Diphoton_phi', 'Diphoton_helicity', 'Diphoton_pt_mgg', 'Diphoton_max_mvaID', 'Diphoton_min_mvaID', 'Diphoton_dPhi', 'LeadPhoton_pt', 'LeadPhoton_eta', 'LeadPhoton_phi', 'LeadPhoton_mass', 'LeadPhoton_mvaID', 'SubleadPhoton_pt', 'SubleadPhoton_eta', 'SubleadPhoton_phi', 'SubleadPhoton_mass', 'SubleadPhoton_mvaID', 'n_electrons', 'n_muons', 'n_taus', 'n_iso_tracks', 'n_jets', 'n_bjets', 'MET_pt', 'MET_phi', 'diphoton_met_dPhi', 'lead_lepton_met_dphi', 'ditau_dphi', 'ditau_deta', 'ditau_dR', 'ditau_mass', 'ditau_pt', 'ditau_eta', 'lead_lepton_pt', 'lead_lepton_eta', 'lead_lepton_phi', 'lead_lepton_mass', 'lead_lepton_charge', 'lead_lepton_id', 'sublead_lepton_pt', 'sublead_lepton_eta', 'sublead_lepton_phi', 'sublead_lepton_mass', 'sublead_lepton_charge', 'sublead_lepton_id', 'category', 'jet_1_pt', 'jet_1_eta', 'jet_1_btagDeepFlavB', 'jet_2_pt', 'jet_2_eta', 'jet_2_btagDeepFlavB', 'b_jet_1_btagDeepFlavB', 'dilep_leadpho_mass', 'dilep_subleadpho_mass', 'year', 'LeadPhoton_genPartFlav', 'SubleadPhoton_genPartFlav'],
  #'additional': ['Diphoton_ditau_dphi', 'LeadPhoton_ditau_dR', 'Diphoton_deta', 'sublead_lepton_met_dPhi', 'Diphoton_sublead_lepton_deta', 'Diphoton_sublead_lepton_dphi', 'Diphoton_ditau_dR', 'Diphoton_ditau_deta', 'reco_MggtauMET', 'LeadPhoton_sublead_lepton_dR', 'ditau_phi', 'SubleadPhoton_lead_lepton_dR', 'Diphoton_lead_lepton_deta', 'Diphoton_lead_lepton_dR', 'Diphoton_dR', 'reco_MX_MET', 'Diphoton_lead_lepton_dphi', 'LeadPhoton_lead_lepton_dR', 'reco_Mggtau', 'ditau_met_dPhi', 'SubleadPhoton_ditau_dR', 'Diphoton_sublead_lepton_dR', 'reco_MX', 'SubleadPhoton_sublead_lepton_dR']
  'additional': ['Diphoton_ditau_dphi', 'LeadPhoton_ditau_dR', 'Diphoton_deta', 'sublead_lepton_met_dPhi', 'Diphoton_sublead_lepton_deta', 'Diphoton_sublead_lepton_dphi', 'Diphoton_ditau_dR', 'Diphoton_ditau_deta', 'LeadPhoton_sublead_lepton_dR', 'ditau_phi', 'SubleadPhoton_lead_lepton_dR', 'Diphoton_lead_lepton_deta', 'Diphoton_lead_lepton_dR', 'Diphoton_dR', 'Diphoton_lead_lepton_dphi', 'LeadPhoton_lead_lepton_dR', 'ditau_met_dPhi', 'SubleadPhoton_ditau_dR', 'Diphoton_sublead_lepton_dR', 'SubleadPhoton_sublead_lepton_dR']
}
train_features['all'] = train_features['base'] + train_features['additional']
train_features['important'] = ["Diphoton_dR", "ditau_mass", "LeadPhoton_pt", "Diphoton_lead_lepton_dphi", "lead_lepton_mass", "SubleadPhoton_genPartFlav", "Diphoton_lead_lepton_dR", "diphoton_met_dPhi", "MET_pt", "Diphoton_sublead_lepton_dR", "LeadPhoton_lead_lepton_dR", "dilep_subleadpho_mass", "SubleadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "lead_lepton_pt", "category", "ditau_met_dPhi", "SubleadPhoton_pt", "Diphoton_deta", "LeadPhoton_genPartFlav", "sublead_lepton_pt", "Diphoton_lead_lepton_deta", "jet_1_pt", "Diphoton_min_mvaID", "Diphoton_sublead_lepton_dphi", "dilep_leadpho_mass", "b_jet_1_btagDeepFlavB", "ditau_pt", "ditau_dR", "Diphoton_dPhi"]

def get_MX_MY(sig_proc):
  if "radion" in sig_proc:
    MX = float(sig_proc.split("M")[1].split("_")[0])
    MY = 125.0
  elif "NMSSM" in sig_proc:
    split_name = sig_proc.split("_")
    MX = float(split_name[7])
    MY = float(split_name[9])
  elif "XToHHggTauTau" in sig_proc:
    MX = float(sig_proc.split("_")[1][1:])
    MY = 125.0
  else:
    raise Exception("Unexpected signal process: %s"%sig_proc)
  return MX, MY

def parserToList(args):
  names = list(filter(lambda x: x[0] != "_", dir(args)))
  l = []
  for name in names:
    value = getattr(args, name)
    if type(value) == list:
      if len(value) > 0: l.append("--%s %s"%(name.replace("_", "-"), " ".join(value)))
    elif value is True:
      l.append("--%s"%name.replace("_", "-"))
    elif type(value) in [str, float, int]:
      l.append("--%s %s"%(name.replace("_", "-"), value))
  return l

def submitToBatch(argv):
  COMMAND = argv
  COMMAND.remove("--batch")
  COMMAND = " ".join(COMMAND)
  
  PWD = os.getcwd()

  with open("batch_template.sh", "r") as f:
    template = f.read()
  filled_template = template % {
    "PWD": PWD,
    "COMMAND": COMMAND
  }

  os.makedirs("batch", exist_ok=True)
  
  script_name = argv[0].replace("/", "_").replace(".", "_")
  existing_files = list(filter(lambda x: script_name in x, os.listdir("batch")))

  script_path = "batch/%s_%d.sh"%(script_name, len(existing_files)+1)
  with open(script_path, "w") as f:
    f.write(filled_template)

  submit_command = "qsub -q hep.q -l h_vmem=24G %s"%script_path
  print(">> Submitting to batch")
  print(submit_command)
  os.system(submit_command)