import os

LOW_MASS_MODE = True
print('\033[93m')
if LOW_MASS_MODE: print("Running in LOW MASS MODE")
else:             print("Running in HIGH MASS MODE")
print('\033[0m')

dummy_val = -9.0 #value used for missing variables, e.g. sublead_lepton_pt when there is only one lepton

if not LOW_MASS_MODE:  
  lumi_table = {
    2016: 36.31,
    2017: 41.48,
    2018: 59.83
  }
else:
  lumi_table = {
    2016: 36.31,
    2017: 41.48,
    2018: 54.67
  }
tot_lumi = "%.3g"%(lumi_table[2016]+lumi_table[2017]+lumi_table[2018])

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
  "X_HH": ['XToHHggTauTau_M260', 'XToHHggTauTau_M270', 'XToHHggTauTau_M280', 'XToHHggTauTau_M290', 'XToHHggTauTau_M300', 'XToHHggTauTau_M320', 'XToHHggTauTau_M350', 'XToHHggTauTau_M400', 'XToHHggTauTau_M450', 'XToHHggTauTau_M500', 'XToHHggTauTau_M550', 'XToHHggTauTau_M600', 'XToHHggTauTau_M650', 'XToHHggTauTau_M700', 'XToHHggTauTau_M750', 'XToHHggTauTau_M800', 'XToHHggTauTau_M900', 'XToHHggTauTau_M1000'],
  "X_YH_Y_gg": ['NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_700', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_800', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_300', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_150', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_200', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_250', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_400', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_500', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_600', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_700', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_90'],
  "X_YH_Y_tautau" : ['NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_700', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_600', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_600', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_700', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_500', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_800_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_600', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_200', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_300', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_600_MY_250', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_400', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_90', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_80', 'NMSSM_XYH_Y_tautau_H_gg_MX_400_MY_150', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_1000_MY_800', 'NMSSM_XYH_Y_tautau_H_gg_MX_300_MY_70', 'NMSSM_XYH_Y_tautau_H_gg_MX_700_MY_100', 'NMSSM_XYH_Y_tautau_H_gg_MX_900_MY_125', 'NMSSM_XYH_Y_tautau_H_gg_MX_500_MY_90'],
  "X_YH_Y_gg_low_mass": ['NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_1000_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_300_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_400_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_500_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_600_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_700_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_800_MY_90', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_100', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_125', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_70', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_80', 'NMSSM_XYH_Y_gg_H_tautau_MX_900_MY_90']
}
sig_procs["all"] = sig_procs["X_HH"] + sig_procs["X_YH_Y_gg"] + sig_procs["X_YH_Y_tautau"]

if not LOW_MASS_MODE:
  bkg_procs = {
    'Diphoton': ['DiPhoton'],
    'GJets': ['GJets_HT-100To200', 'GJets_HT-200To400', 'GJets_HT-400To600', 'GJets_HT-40To100', 'GJets_HT-600ToInf'],
    'TT': ['TTGG', 'TTGamma', 'TTJets'],
    'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
    'VGamma': ['WGamma', 'ZGamma'],
  }
else:
  bkg_procs = {
    #'Diphoton': ['DiPhoton40to80','DiPhoton'],
    'Diphoton': ['DiPhoton'],
    'GJets': ['GJets_HT-100To200', 'GJets_HT-200To400', 'GJets_HT-400To600', 'GJets_HT-40To100', 'GJets_HT-600ToInf'],
    'TT': ['TTGG', 'TTGamma', 'TTJets'],
    'SM Higgs': ['VBFH_M125', 'VH_M125', 'ggH_M125', 'ttH_M125'],
    'VGamma': ['WGamma', 'ZGamma'],
    #'DY': ['DY'],
    #'VV': ['WW', 'ZZ', 'WZ']
  }
bkg_procs["all"] = [proc for key in bkg_procs.keys() for proc in bkg_procs[key]]

weights_systematics = ['weight_tau_idDeepTauVSe_sf_AnalysisTau_central', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_down', 'weight_muon_id_sfSTAT_SelectedMuon_down', 'weight_central_initial', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_up', 'weight_muon_iso_sfSTAT_SelectedMuon_down', 'weight_muon_id_sfSYS_SelectedMuon_central', 'weight_muon_iso_sfSYS_SelectedMuon_up', 'weight_electron_id_sf_SelectedElectron_up', 'weight_puWeight_up', 'weight_electron_id_sf_SelectedElectron_central', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_up', 'weight_muon_iso_sfSYS_SelectedMuon_down', 'weight_puWeight_down', 'weight_electron_id_sf_SelectedElectron_down', 'weight_central_no_lumi', 'weight_muon_id_sfSYS_SelectedMuon_up', 'weight_muon_id_sfSTAT_SelectedMuon_central', 'weight_tau_idDeepTauVSjet_sf_AnalysisTau_central', 'weight_trigger_sf_central', 'weight_electron_veto_sf_Diphoton_Photon_central', 'weight_muon_iso_sfSTAT_SelectedMuon_central', 'weight_L1_prefiring_sf_up', 'weight_electron_veto_sf_Diphoton_Photon_down', 'weight_muon_id_sfSTAT_SelectedMuon_up', 'weight_electron_veto_sf_Diphoton_Photon_up', 'weight_muon_iso_sfSYS_SelectedMuon_central', 'weight_puWeight_central', 'weight_L1_prefiring_sf_down', 'weight_muon_id_sfSYS_SelectedMuon_down', 'weight_muon_iso_sfSTAT_SelectedMuon_up', 'weight_trigger_sf_down', 'weight_trigger_sf_up', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_down', 'weight_L1_prefiring_sf_central', 'weight_tau_idDeepTauVSe_sf_AnalysisTau_down', 'weight_tau_idDeepTauVSmu_sf_AnalysisTau_central', 'weight_tau_idDeepTauVSe_sf_AnalysisTau_up']

train_features = {
  'base': ['Diphoton_eta', 'Diphoton_phi', 'Diphoton_helicity', 'Diphoton_pt_mgg', 'Diphoton_max_mvaID', 'Diphoton_min_mvaID', 'Diphoton_dPhi', 'LeadPhoton_pt_mgg', 'LeadPhoton_eta', 'LeadPhoton_phi', 'LeadPhoton_mvaID', 'SubleadPhoton_pt_mgg', 'SubleadPhoton_eta', 'SubleadPhoton_phi', 'SubleadPhoton_mvaID', 'n_electrons', 'n_muons', 'n_taus', 'n_iso_tracks', 'n_jets', 'n_bjets', 'MET_pt', 'MET_phi', 'diphoton_met_dPhi', 'lead_lepton_met_dphi', 'ditau_dphi', 'ditau_deta', 'ditau_dR', 'ditau_mass', 'ditau_pt', 'ditau_eta', 'lead_lepton_pt', 'lead_lepton_eta', 'lead_lepton_phi', 'lead_lepton_mass', 'lead_lepton_charge', 'lead_lepton_id', 'sublead_lepton_pt', 'sublead_lepton_eta', 'sublead_lepton_phi', 'sublead_lepton_mass', 'sublead_lepton_charge', 'sublead_lepton_id', 'category', 'jet_1_pt', 'jet_1_eta', 'jet_1_btagDeepFlavB', 'jet_2_pt', 'jet_2_eta', 'jet_2_btagDeepFlavB', 'b_jet_1_btagDeepFlavB', 'dilep_leadpho_mass', 'dilep_subleadpho_mass', 'year'],
  'additional': ['reco_MggtauMET_mgg', 'reco_MX', 'reco_MX_MET', 'Diphoton_deta', 'Diphoton_ditau_deta', 'LeadPhoton_lead_lepton_dR', 'Diphoton_lead_lepton_dR', 'LeadPhoton_sublead_lepton_dR', 'Diphoton_sublead_lepton_deta', 'Diphoton_sublead_lepton_dphi', 'Diphoton_lead_lepton_deta', 'SubleadPhoton_lead_lepton_dR', 'Diphoton_lead_lepton_dphi', 'LeadPhoton_ditau_dR', 'SubleadPhoton_ditau_dR', 'SubleadPhoton_sublead_lepton_dR', 'sublead_lepton_met_dPhi', 'ditau_met_dPhi', 'Diphoton_dR', 'Diphoton_sublead_lepton_dR', 'ditau_phi', 'Diphoton_ditau_dphi', 'Diphoton_ditau_dR']
}
train_features['all'] = train_features['base'] + train_features['additional']
train_features['important_corr'] = ["Diphoton_dR", "LeadPhoton_sublead_lepton_dR", "ditau_pt", "jet_2_btagDeepFlavB", "lead_lepton_mass", "LeadPhoton_ditau_dR", "lead_lepton_pt", "dilep_subleadpho_mass", "ditau_mass", "lead_lepton_charge", "dilep_leadpho_mass", "Diphoton_helicity", "LeadPhoton_pt_mgg", "MET_pt", "reco_MX_MET", "Diphoton_lead_lepton_dR", "ditau_met_dPhi", "b_jet_1_btagDeepFlavB", "sublead_lepton_id", "Diphoton_ditau_dR", "n_bjets", "SubleadPhoton_sublead_lepton_dR", "category", "ditau_dR", "jet_1_btagDeepFlavB", "jet_1_pt", "diphoton_met_dPhi", "Diphoton_pt_mgg", "reco_MggtauMET_mgg", "lead_lepton_met_dphi", "SubleadPhoton_pt_mgg", "sublead_lepton_pt", "SubleadPhoton_lead_lepton_dR", "LeadPhoton_lead_lepton_dR", "Diphoton_sublead_lepton_dR"]
train_features['important_30'] = ["Diphoton_sublead_lepton_deta", "Diphoton_dR", "Diphoton_sublead_lepton_dR", "LeadPhoton_sublead_lepton_dR", "ditau_pt", "jet_2_btagDeepFlavB", "lead_lepton_mass", "LeadPhoton_ditau_dR", "lead_lepton_pt", "n_electrons", "Diphoton_ditau_dphi", "dilep_subleadpho_mass", "ditau_mass", "Diphoton_sublead_lepton_dphi", "lead_lepton_charge", "dilep_leadpho_mass", "Diphoton_helicity", "ditau_deta", "n_taus", "Diphoton_lead_lepton_dphi", "LeadPhoton_pt_mgg", "MET_pt", "reco_MX_MET", "Diphoton_lead_lepton_dR", "ditau_met_dPhi", "b_jet_1_btagDeepFlavB", "sublead_lepton_id", "Diphoton_ditau_dR", "n_bjets", "Diphoton_lead_lepton_deta", "SubleadPhoton_sublead_lepton_dR", "category", "ditau_dR", "jet_1_btagDeepFlavB", "reco_MX", "jet_1_pt", "sublead_lepton_eta", "ditau_dphi", "diphoton_met_dPhi", "Diphoton_pt_mgg", "lead_lepton_eta", "reco_MggtauMET_mgg", "lead_lepton_met_dphi", "Diphoton_ditau_deta", "SubleadPhoton_pt_mgg", "sublead_lepton_pt", "SubleadPhoton_lead_lepton_dR", "n_iso_tracks", "Diphoton_dPhi", "LeadPhoton_lead_lepton_dR"]
train_features['important_20'] = ["Diphoton_helicity", "ditau_deta", "Diphoton_lead_lepton_deta", "category", "SubleadPhoton_sublead_lepton_dR", "SubleadPhoton_lead_lepton_dR", "LeadPhoton_lead_lepton_dR", "lead_lepton_mass", "SubleadPhoton_pt_mgg", "lead_lepton_pt", "ditau_mass", "LeadPhoton_pt_mgg", "MET_pt", "Diphoton_lead_lepton_dR", "ditau_dR", "Diphoton_pt_mgg", "reco_MX_MET", "ditau_pt", "ditau_met_dPhi", "Diphoton_ditau_dphi", "Diphoton_dPhi", "Diphoton_lead_lepton_dphi", "Diphoton_dR", "Diphoton_sublead_lepton_dR", "reco_MX", "ditau_dphi", "jet_1_pt", "LeadPhoton_ditau_dR", "reco_MggtauMET_mgg", "Diphoton_sublead_lepton_deta", "diphoton_met_dPhi", "dilep_leadpho_mass", "Diphoton_sublead_lepton_dphi", "Diphoton_ditau_deta"]
train_features['important_15'] = ["Diphoton_dR", "Diphoton_lead_lepton_dphi", "diphoton_met_dPhi", "SubleadPhoton_pt_mgg", "reco_MggtauMET_mgg", "ditau_dphi", "ditau_mass", "Diphoton_lead_lepton_dR", "lead_lepton_pt", "Diphoton_lead_lepton_deta", "LeadPhoton_pt_mgg", "SubleadPhoton_lead_lepton_dR", "Diphoton_ditau_deta", "reco_MX", "ditau_pt", "ditau_dR", "Diphoton_sublead_lepton_dphi", "dilep_leadpho_mass", "MET_pt", "Diphoton_pt_mgg", "LeadPhoton_lead_lepton_dR", "ditau_met_dPhi", "LeadPhoton_ditau_dR", "jet_1_pt", "category", "Diphoton_dPhi", "reco_MX_MET", "Diphoton_sublead_lepton_dR", "Diphoton_sublead_lepton_deta", "ditau_deta"]
train_features['important_16'] = ["LeadPhoton_ditau_dR", "dilep_leadpho_mass", "ditau_mass", "Diphoton_lead_lepton_deta", "ditau_dphi", "Diphoton_sublead_lepton_dR", "ditau_dR", "Diphoton_ditau_deta", "SubleadPhoton_lead_lepton_dR", "reco_MggtauMET_mgg", "reco_MX_MET", "Diphoton_dR", "category", "LeadPhoton_lead_lepton_dR", "lead_lepton_pt", "MET_pt", "Diphoton_pt_mgg", "reco_MX", "ditau_pt", "ditau_deta", "SubleadPhoton_pt_mgg", "Diphoton_dPhi", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "Diphoton_lead_lepton_dphi", "LeadPhoton_pt_mgg", "diphoton_met_dPhi", "Diphoton_sublead_lepton_deta", "Diphoton_sublead_lepton_dphi", "jet_1_pt"]
train_features['important_17'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "Diphoton_lead_lepton_dphi", "ditau_pt", "category", "reco_MX", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "Diphoton_dR", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "reco_MggtauMET_mgg", "Diphoton_sublead_lepton_dphi", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['important_17_corr'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "reco_MggtauMET_mgg", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]
train_features['important_17_corr_no_mggtau'] = ["LeadPhoton_pt_mgg", "ditau_mass", "SubleadPhoton_lead_lepton_dR", "ditau_dR", "Diphoton_dPhi", "ditau_deta", "LeadPhoton_lead_lepton_dR", "Diphoton_pt_mgg", "reco_MX_MET_mgg", "Diphoton_ditau_deta", "lead_lepton_mass", "diphoton_met_dPhi", "ditau_pt", "category", "Diphoton_sublead_lepton_dR", "MET_pt", "jet_1_pt", "ditau_dphi", "dilep_leadpho_mass", "lead_lepton_pt", "ditau_met_dPhi", "Diphoton_lead_lepton_dR", "LeadPhoton_ditau_dR", "Diphoton_lead_lepton_deta", "Diphoton_sublead_lepton_deta", "SubleadPhoton_pt_mgg", "Diphoton_ditau_dphi"]

def get_MX_MY(sig_proc):
  if "NMSSM" in sig_proc:
    split_name = sig_proc.split("_")
    MX = float(split_name[7])
    MY = float(split_name[9])
  elif "XToHHggTauTau" in sig_proc:
    MX = float(sig_proc.split("_")[1][1:])
    MY = 125.0
  else:
    raise Exception("Unexpected signal process: %s"%sig_proc)
  return MX, MY

def get_sig_proc(example_sig_proc, mx, my):
  if "XToHHggTauTau" in example_sig_proc:
    return "XToHHggTauTau_M%d"%mx
  elif "NMSSM_XYH_Y_gg_H_tautau" in example_sig_proc:
    return "NMSSM_XYH_Y_gg_H_tautau_MX_%d_MY_%d"%(mx, my)
  elif "NMSSM_XYH_Y_tautau_H_gg" in example_sig_proc:
    return "NMSSM_XYH_Y_tautau_H_gg_MX_%d_MY_%d"%(mx, my)

def add_MX_MY(df, proc_dict):
  #get sig_procs in parquet file
  sig_procs_in = [sig_proc for sig_proc in sig_procs["all"] if sig_proc in proc_dict.keys()]

  df["MX"] = dummy_val
  df["MY"] = dummy_val
  
  for sig_proc in sig_procs_in:
    MX, MY = get_MX_MY(sig_proc)
    
    df.loc[df.process_id==proc_dict[sig_proc], "MX"] = MX
    df.loc[df.process_id==proc_dict[sig_proc], "MY"] = MY

def parserToList(args):
  names = list(filter(lambda x: x[0] != "_", dir(args)))
  #print(names)
  l = []
  for name in names:
    value = getattr(args, name)
    if type(value) in [list, tuple]:
      if len(value) > 0: l.append("--%s %s"%(name.replace("_", "-"), " ".join([str(v) for v in value])))
    elif value is True:
      l.append("--%s"%name.replace("_", "-"))
    elif type(value) in [str, float, int]:
      l.append("--%s %s"%(name.replace("_", "-"), value))
    elif value is False:
      pass
    elif value is None:
      pass
    else:
      print("Dont know what to do with %s of type %s"%(name, type(value)))

  l_split = []
  for each in l:
    l_split.extend(each.split(" "))
  return l_split

def submitToBatch(argv, extra_memory=False, quick=False):
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

  if type(extra_memory) == bool:
    if extra_memory: slots = 2
    else:            slots = 1
  else:
    slots = extra_memory

  if not quick:  submit_command = "qsub -q hep.q -l h_vmem=24G -l h_rt=3:0:0 -pe hep.pe %d %s"%(slots, script_path)
  else:          submit_command = "qsub -q hep.q -l h_vmem=24G -l h_rt=600 %s"%script_path
  os.system(submit_command)

from pyarrow.parquet import ParquetFile
def getColumns(parquet_file):
  pf = ParquetFile(parquet_file) 
  columns = [each.name for each in pf.schema]
  columns.remove("__index_level_0__")
  return columns

import json
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)