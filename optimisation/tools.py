import common
import math

def get_pres(sig_proc):
  """Return the correct preselection range for this sig proc"""
  if ("XToHHggTauTau" in sig_proc) or ("HHTo2G2Tau" in sig_proc): #if graviton
    return (100, 180)
  elif "NMSSM_XYH_Y_gg_H_tautau" in sig_proc: #if Y_gg
    if common.LOW_MASS_MODE:
      return (65, 150)
    else:
      return (100, 1000)
  elif "NMSSM_XYH_Y_tautau_H_gg" in sig_proc: #if Y_tautau
    return (100, 180)

def get_sr(sig_proc):
  """Return the correct signal region range for this sig proc"""
  if ("XToHHggTauTau" in sig_proc) or ("HHTo2G2Tau" in sig_proc): #if graviton
    return (115, 135)
  elif "NMSSM_XYH_Y_gg_H_tautau" in sig_proc: #if Y_gg
    mx, my = common.get_MX_MY(sig_proc)
    
    width = math.ceil(10 * (my/125.))
    low, high = my-width, my+width
    low = max([low, 68]) #don't let lower bound go lower than 68 GeV
    return (low, high)

  elif "NMSSM_XYH_Y_tautau_H_gg" in sig_proc: #if Y_tautau
    return (115, 135)