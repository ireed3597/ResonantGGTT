from random import sample
import numpy as np
from sklearn.metrics import roc_curve

def getAUC(fpr, tpr):
  """
  Calculate AUC score. Works for negative weights.
  When you have negative weights in the background, the roc curve can go 
  back on itself which can lead to AUC>1 when integrating over this curve.
  To fix this, the algorithm 'skips' over the parts of the curve where it
  goes back on itself.
  """

  fixed_fpr = []
  fixed_tpr = []

  last_fpr = -1
  for i in range(len(fpr)):
    if fpr[i] >= last_fpr:
      fixed_fpr.append(fpr[i])
      fixed_tpr.append(tpr[i])
      last_fpr = fpr[i]

  return np.trapz(fixed_tpr, fixed_fpr)
  
if __name__=="__main__":
  n=100000
  w_choice = [-100,-1,1,1]

  sig_score = np.random.random(size=n)**0.5
  sig_y = np.ones_like(sig_score)
  sig_w = np.random.choice(w_choice, size=n, p=[0.001,0.009,0.98,0.01])

  bkg_score = np.random.random(size=n)**2
  bkg_y = np.zeros_like(bkg_score)
  bkg_w = np.random.choice(w_choice, size=n, p=[0.001,0.009,0.98,0.01])

  score = np.concatenate([sig_score, bkg_score])
  y = np.concatenate([sig_y, bkg_y])
  w = np.concatenate(([sig_w, bkg_w]))

  fpr, tpr, thresholds = roc_curve(y, score, sample_weight=w)

  print("Just integrate: %.8f"%(np.trapz(tpr, fpr)))
  print("Skip negative parts: %.8f"%getAUC(fpr, tpr))

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  plt.plot(fpr, tpr)
  plt.savefig("auc_test.pdf")