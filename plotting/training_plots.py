import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep
mplhep.set_style("CMS")
plt.rcParams["figure.figsize"] = (12.5,10)

import pandas as pd
import numpy as np

import os

import json

from plotting.plot_input_features import plot_feature
from training.auc import getAUC

def plotOutputScore(data, sig, bkg, proc_dict, sig_proc, savein):
  #bkg_rw = bkg.copy()
  #bkg_rw.loc[:, "weight"] *= data.loc[:, "weight"].sum() / bkg_rw.loc[:, "weight"].sum()

  for column in data.columns:
    if ("score" in column) & (sig_proc in column):
      plot_feature(data, bkg, sig, proc_dict, sig_proc, column, 25, (0,1), os.path.join(savein, column))
      plot_feature(data, bkg, sig, proc_dict, sig_proc, column, 25, (0.99,1), os.path.join(savein, column+"_zoom"))
      #plot_feature(data, bkg, sig, proc_dict, sig_proc, column, 50, (0.999,1), os.path.join(savein, column+"_zoom2"))
      #plot_feature(data, bkg_rw, sig, proc_dict, sig_proc, column, 50, (0,1), os.path.join(savein, column+"_bkg_normed"))

def plotROC(train_fpr, train_tpr, test_fpr, test_tpr, savein):
  #train_auc = np.trapz(train_tpr, train_fpr)
  #test_auc = np.trapz(test_tpr, test_fpr)
  train_auc = getAUC(train_fpr, train_tpr)
  test_auc = getAUC(test_fpr, test_tpr)

  # wanted_fpr = np.linspace(0, 1, 1001) #save slices in roc curve
  # save_package = {
  #   "train_auc": train_auc,
  #   "test_auc": test_auc,
  #   "train_fpr": list(train_fpr[np.searchsorted(train_fpr, wanted_fpr)]),
  #   "train_tpr": list(train_tpr[np.searchsorted(train_fpr, wanted_fpr)]),
  #   "test_fpr": list(test_fpr[np.searchsorted(test_fpr, wanted_fpr)]),
  #   "test_tpr": list(test_tpr[np.searchsorted(test_fpr, wanted_fpr)])
  # }
  save_package = {
    "train_auc": train_auc,
    "test_auc": test_auc,
    "train_fpr": list(train_fpr),
    "train_tpr": list(train_tpr),
    "test_fpr": list(test_fpr),
    "test_tpr": list(test_tpr)
  }
  with open(os.path.join(savein, "ROC.json"), "w") as f:
    json.dump(save_package, f, indent=4)

  save_package = {
    "train_auc": train_auc,
    "test_auc": test_auc,
  }
  with open(os.path.join(savein, "ROC_skimmed.json"), "w") as f:
    json.dump(save_package, f, indent=4)

  plt.plot(train_fpr, train_tpr, label="Train AUC = %.4f"%train_auc)
  plt.plot(test_fpr, test_tpr, label="Test AUC = %.4f"%test_auc)
  plt.xlabel("False positive rate")
  plt.ylabel("True positive rate")
  plt.legend()
  plt.savefig(os.path.join(savein, "ROC.png"))
  # plt.xlim(left=0.1)
  # plt.ylim(bottom=min(test_tpr[test_fpr>0.1]), top=1+(1-min(test_tpr[test_fpr>0.1]))*0.1)
  # plt.savefig(os.path.join(savein, "ROC_zoom.png"))
  # plt.xscale("log")
  # plt.savefig(os.path.join(savein, "ROC_log.png"))
  plt.clf()

def plotLoss(train_loss, validation_loss, savein):
  n_epochs = len(train_loss)

  plt.plot(np.arange(n_epochs), train_loss, label="Train")
  plt.plot(np.arange(n_epochs), validation_loss, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig(os.path.join(savein, "loss.png"))
  plt.clf()