import torch
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import random
import os

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import training.custom_modules as cm
from torchviz import make_dot

import xgboost as xgb

import tracemalloc

print(torch.cuda.is_available())
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu" 

def setSeed(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)

class Model:
  def __init__(self, hyperparams=None):
    self.initModel(hyperparams)

  def printNumAndWeight(self, y, w):
    print(" nsig = %d"%sum(y==1))
    print(" nbkg = %d"%sum(y==0))
    print(" sum wsig = %f"%sum(w[y==1]))
    print(" sum wbkg = %f"%sum(w[y==0]))

  def equaliseWeights(self, X, y, w):
    w[y==1] *= w[y==0].sum() / w[y==1].sum()
    assert np.isclose(w[y==0].sum(), w[y==1].sum()), print("Equalisation of weights failed. \nBkg sumw = %f \nSig sumw = %f"%(w[y==0].sum(), w[y==1].sum()))


class BDT(Model):
  def initModel(self, hyperparams=None):
    if hyperparams == None: hyperparams={'objective':'binary:logistic', 'n_estimators':100, 'eta':0.05, 'max_depth':4, 'subsample':0.6, 'colsample_bytree':0.6, 'gamma':1}
    # self.model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, 
    #                                 eta=0.05, max_depth=4,
    #                                 subsample=0.6, colsample_bytree=0.6, gamma=1)
    #self.model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=500, 
    #                                 eta=0.05, max_depth=4,
    #                                 subsample=0.6, colsample_bytree=0.6, gamma=1)
    self.model = xgb.XGBClassifier(**hyperparams)    

  def fit(self, X, y, w):
    self.equaliseWeights(X, y, w)
    print(">> Training sample summary")
    self.printNumAndWeight(y, w)
    self.model.fit(X, y, sample_weight=w)

  def predict_proba(self, X):
    return self.model.predict_proba(X)

  def score(self, X, y, sample_weight=None):
    return self.model.score(X, y, sample_weight)


class ParamModel(Model):
  def __init__(self, n_params, n_sig_procs, n_features, hyperparams=None):
    self.n_params = n_params
    self.n_sig_procs = n_sig_procs
    self.n_features = n_features
    self.initModel(hyperparams)

  def equaliseWeights(self, X, y, w):
    #In ParamModel we first equalise weights among signal processes
    
    #find unique combinations of parameters (masses)
    unique_combinations, counts = np.unique(X[y==1,-self.n_params:], axis=0, return_counts=True)

    norm = counts[0] #arbitarily choose the number of events from first signal processes to norm to
    #norm = 1

    for combination in unique_combinations:
      signal_proc_selection = (X[:,-self.n_params:] == combination).all(axis=1) & y==1
      w[signal_proc_selection] *= norm / w[signal_proc_selection].sum() #norm to sumw = 1
    assert np.isclose(w[y==1].sum(), norm*self.n_sig_procs), print("Equalisation amongst signal processes failed. \nn_sig_procs = %d \nsig_sum_w = %f"%(self.n_sig_procs, w[y==1].sum()))

    return Model.equaliseWeights(self, X, y, w)

  def shuffleBkg(self, X, y):
    """Randomly assign values of possible parameters (masses) to the background"""
    #find unique combinations of parameters (masses)
    unique_combinations = np.unique(X[y==1,-self.n_params:], axis=0)

    choices = np.random.choice(np.arange(len(unique_combinations)), sum(y==0))
    X[y==0,-self.n_params:] = unique_combinations[choices]   
    return X

  def inflateBkgWithMasses(self, X, y, w):
    print("Bkg inflate 1", tracemalloc.get_traced_memory())

    X_sig, y_sig, w_sig = X[y==1], y[y==1], w[y==1]
    X_bkg, y_bkg, w_bkg = X[y==0], y[y==0], w[y==0]
    print("Bkg inflate 2", tracemalloc.get_traced_memory())
    
    #lists to hold signal samples and copies of bkg before concatenating
    Xs, ys, ws = [X_sig], [y_sig], [w_sig]
    print("Bkg inflate 3", tracemalloc.get_traced_memory())  
    
    #find unique combinations of parameters (masses)
    unique_combinations = np.unique(X[y==1,-self.n_params:], axis=0)
    for combination in unique_combinations:
      X_bkg_c, y_bkg_c, w_bkg_c = X_bkg.copy(), y_bkg.copy(), w_bkg.copy()
      X_bkg_c[:, -self.n_params:] = combination
      print("Bkg inflate 4", tracemalloc.get_traced_memory())
      
      Xs.append(X_bkg_c)
      ys.append(y_bkg_c)
      ws.append(w_bkg_c)
      print("Bkg inflate 5", tracemalloc.get_traced_memory())

    X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
    print("Bkg inflate 6", tracemalloc.get_traced_memory())
    return X, y, w

class ParamBDT(ParamModel):
  def initModel(self, hyperparams=None):
    if hyperparams == None: hyperparams={'objective':'binary:logistic', 'n_estimators':100, 'eta':0.05, 'max_depth':4, 'subsample':0.6, 'colsample_bytree':0.6, 'gamma':1}
    #self.model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=200, 
    #                               eta=0.05, max_depth=4,
    #                               subsample=0.6, colsample_bytree=0.6, gamma=1)
    #self.model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=200,
    #                                eta=0.3, maxDepth=9, min_child_weight=0.01,
    #                                subsample=1, colsample_bytree=0.6, gamma=0)
    self.model = xgb.XGBClassifier(**hyperparams)

  def fit(self, X, y, w):
    #X = self.shuffleBkg(X, y)
    X, y, w = self.inflateBkgWithMasses(X, y, w)
    self.equaliseWeights(X, y, w)
    print(">> Training sample summary")
    self.printNumAndWeight(y, w)
    self.model.fit(X, y, sample_weight=w)

  def predict_proba(self, X):
    #find unique combinations of parameters (masses)
    unique_combinations = np.unique(X[:,-self.n_params:], axis=0)
    assert len(unique_combinations)==1, print("Expect only one combination of parameters (masses) when evaluating ParamBDT")

    return self.model.predict_proba(X)

class ParamNN(ParamModel):
  def initModel(self, hyperparams=None):
    self.outdir = None
    self.model_save_name = None

    if hyperparams==None:
      self.hyperparams = {
        "max_epochs": 100,
        "batch_size": 128,
        "lr": 0.01,
        #"min_epoch": 20,
        #"grace_epochs": 15,
        "min_epoch": 5,
        "grace_epochs": 5,
        "tol": 0.01,
        "gamma": 0.9,
        "dropout": 0,
        "n_layers": 2,
        "n_nodes": 10
      }
    else:
      self.hyperparams = hyperparams
    
    if self.hyperparams["pass_through"] == 0:
      modules = [
        torch.nn.Linear(self.n_features,self.hyperparams["n_nodes"]),
        torch.nn.Dropout(self.hyperparams["dropout"]),
        torch.nn.ELU()
        ]
      for i in range(self.hyperparams["n_layers"]-1):
        middle_layer = [
          torch.nn.Linear(self.hyperparams["n_nodes"],self.hyperparams["n_nodes"]),
          torch.nn.Dropout(self.hyperparams["dropout"]),
          torch.nn.ELU()
        ]
        modules.extend(middle_layer)
    else:
      modules = [
        cm.PassThroughLayer(self.n_features,self.hyperparams["n_nodes"],self.hyperparams["dropout"]),
        torch.nn.Linear(self.hyperparams["n_nodes"],self.hyperparams["n_nodes"]),
        torch.nn.Dropout(self.hyperparams["dropout"]),
        torch.nn.ELU()
        #cm.PassThroughLayer(self.hyperparams["n_nodes"],self.hyperparams["n_nodes"],self.hyperparams["dropout"]),
        ]
    
    last_layer = [
      torch.nn.Linear(self.hyperparams["n_nodes"],1),
      torch.nn.Flatten(0,1),
      torch.nn.Sigmoid()
    ]
    modules.extend(last_layer)
    
    self.model = torch.nn.Sequential(*modules)

  def outputONNX(self, batch):
    print(torch.onnx.export(self.model, batch, "nn.onnx", ["Training features"], ["Score"]))

  def printParameters(self):
    for name, param in self.model.named_parameters():
      if param.requires_grad:
          print(name, param.data)

  def BCELoss(self, input, target, weight):
    x, y, w = input, target, weight
    log = lambda x: torch.log(x*(1-1e-8) + 1e-8)
    return torch.mean(-w * (y*log(x) + (1-y)*log(1-x)))

  def MSELoss(self, input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

  def getTotLoss(self, X, y, w, batch_size=None, s=None):
    losses = []
    for batch_X, batch_y, batch_w in self.getBatches(X, y, w, batch_size):
      loss = self.BCELoss(self.model(batch_X), batch_y, batch_w)
      losses.append(loss.item())
    return sum(losses)

    # if s==None:
    #   X_torch = torch.tensor(X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
    #   y_torch = torch.tensor(y, dtype=torch.float).to(dev)
    #   w_torch = torch.tensor(w, dtype=torch.float).to(dev)
    # else:
    #   X_torch = torch.tensor(X[s], dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
    #   y_torch = torch.tensor(y[s], dtype=torch.float).to(dev)
    #   w_torch = torch.tensor(w[s], dtype=torch.float).to(dev)
    # return self.BCELoss(self.model(X_torch), y_torch, w_torch)

  def getBatches(self, X, y, w, batch_size=None, shuffle=False, weighted=False, epoch_size=None):
    if epoch_size==None: epoch_size = len(X)
    if batch_size==None: batch_size = len(X)

    if shuffle and not weighted:
      shuffle_ids = np.random.choice(len(X), epoch_size, replace=False)
      X_sh = X[shuffle_ids].copy()
      y_sh = y[shuffle_ids].copy()
      w_sh = w[shuffle_ids].copy()
    elif weighted:
      if (not hasattr(self, "normed_weights")) or (len(w) != len(self.normed_weights)): self.normed_weights = abs(w)/sum(abs(w))
      weighted_ids = np.random.choice(len(X), epoch_size, replace=True, p=self.normed_weights)
      X_sh = X[weighted_ids].copy()
      y_sh = y[weighted_ids].copy()
      w_sh = w[weighted_ids].copy()
      w_sh[w_sh>0] = 1
      w_sh[w_sh<0] = -1
    else:
      X_sh = X.copy()
      y_sh = y.copy()
      w_sh = w.copy()  
    
    for i_picture in range(0, epoch_size, batch_size):
      batch_X = X_sh[i_picture:i_picture + batch_size]
      batch_y = y_sh[i_picture:i_picture + batch_size]
      batch_w = w_sh[i_picture:i_picture + batch_size]
    
      X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      y_torch = torch.tensor(batch_y, dtype=torch.float).to(dev)
      w_torch = torch.tensor(batch_w, dtype=torch.float).to(dev)

      yield X_torch, y_torch, w_torch

  def shouldEarlyStop(self):
    """
    Want to stop if seeing no appreciable improvment.
    Check 1. Was the best score more than grace_epochs epochs ago?
          2. If score is improving, has it improved by more than
             tol percent over grace_epochs?
    """

    n_epochs = len(self.validation_loss)

    if n_epochs < self.hyperparams["min_epoch"]:
      return False

    losses = np.array(self.validation_loss)
    slosses = losses.sum(axis=1)

    # #check to see if loss unstable
    # if n_epochs > grace_epochs:
    #   any_unstable = False
    #   for i in range(len(self.train_masses)):
    #     best_loss = losses[:,i].min()        
    #     variation = losses[:,i][-grace_epochs:].max() - losses[:,i][-grace_epochs:].min()
    #     #print(variation, variation/best_loss)
    #     if variation/best_loss > tol:
    #       any_unstable = True
    #       break
    #   if any_unstable:
    #     return False

    #check if best loss happened a while ago
    best_loss = slosses.min()
    best_loss_epoch = np.where(slosses==best_loss)[0][0] + 1

    if n_epochs - best_loss_epoch > self.hyperparams["grace_epochs"]:
      print("Best loss happened a while ago")
      return True

    #check to see if any mass points making good progress
    if n_epochs > self.hyperparams["grace_epochs"]:
      any_make_good_improvement = False

      #first check the total loss improvement
      best_loss = slosses.min()        
      best_loss_before = slosses[:-self.hyperparams["grace_epochs"]].min()
      if (best_loss_before-best_loss)/best_loss > self.hyperparams["tol"]:
        any_make_good_improvement = True

      for i in range(len(losses[0])):
        best_loss = losses[:,i].min()        
        best_loss_before = losses[:,i][:-self.hyperparams["grace_epochs"]].min()
        if (best_loss_before-best_loss)/best_loss > self.hyperparams["tol"]:
          any_make_good_improvement = True
          break
      if not any_make_good_improvement:
        print("Not enough improvement")
        return True

    return False

  def shouldSchedulerStep(self):
    # n_epochs = len(losses)
    
    # losses = np.array(losses)
    # slosses = losses.sum(axis=1)
    # best_loss = slosses.min()
    # best_loss_epoch = np.where(slosses==best_loss)[0][0] + 1

    # if ((n_epochs - best_loss_epoch) > 10) and ((n_epochs - self.last_step_epoch) > 10):
    #   self.last_step_epoch = n_epochs
    #   return True
    # else:
    #   return False

    #return False

    if len(self.train_loss) < 2: return False
    losses = np.array(self.train_loss)
    slosses = losses.sum(axis=1)
    return slosses[-1] > slosses[-2]

  def updateLossPlot(self, train_loss, test_loss, lr):
    if not self.alreadyPlotting:
      plt.ion()
      self.figure, self.ax = plt.subplots()
      x = [i for i in range(1, len(train_loss)+1)]
      self.train_line, = self.ax.plot(x, train_loss, label="train")
      self.test_line, = self.ax.plot(x, test_loss, label="test")
      self.lr_text = self.ax.text(0.1, 1.05, "lr = %f"%lr, transform=self.ax.transAxes)
      self.step_text = self.ax.text(0.5, 1.05, "last scheduler step at epoch %d"%self.last_step_epoch, transform=self.ax.transAxes)
      plt.xlabel("epoch")
      plt.ylabel("loss")
      plt.legend()
      self.alreadyPlotting = True
    else:
      x = [i for i in range(1, len(train_loss)+1)]
      self.train_line.set_data(x, train_loss)
      self.test_line.set_data(x, test_loss)
      self.lr_text.set_text("lr = %f"%lr)
      self.step_text.set_text("last scheduler step at epoch %d"%self.last_step_epoch)
      self.ax.relim()
      self.ax.autoscale_view()
      self.figure.canvas.draw()    
      self.figure.canvas.flush_events()

  def setOutdir(self, outdir):
    self.outdir = outdir

  def saveModel(self):
    if self.model_save_name != None:
      torch.save(self.model, "%s/%s.pt"%(self.outdir, self.model_save_name))
    else:
      #existing_model_names = list(filter(lambda x: ".pt" in x, os.listdir(self.outdir)))
      #self.model_save_name = "model_%d"%(len(existing_model_names))
      self.model_save_name = "model_%d"%int(self.train_loss[0].sum()*10**6) #something probably unique
      self.saveModel()

  def loadModel(self):
    self.model = torch.load("%s/%s.pt"%(self.outdir, self.model_save_name))

  def fit(self, X, y, w):
    X, y, w = self.inflateBkgWithMasses(X, y, w)

    self.unique_combinations = np.unique(X[:,-self.n_params:], axis=0) #unique combinations of masses (MX and MY)
    
    #split samples into training and validation
    Xt, Xv, yt, yv, wt, wv = train_test_split(X, y, w, test_size=0.2, random_state=1)
    assert len(np.unique(Xt[:,-self.n_params:], axis=0)) == len(self.unique_combinations)
    assert len(np.unique(Xv[:,-self.n_params:], axis=0)) == len(self.unique_combinations)

    self.equaliseWeights(Xt, yt, wt)
    self.equaliseWeights(Xv, yv, wv)
    wv *= sum(wt) / sum(wv) #adjust weight of validation to allow comparison of losses
    
    print("Minimum weight in training set: ", min(abs(wt[yt==0])))
    print("Maximum weight in training set: ", max(abs(wt[yt==0])))

    print(">> Training sample summary")
    self.printNumAndWeight(yt, wt)
    print(">> Validation sample summary")
    self.printNumAndWeight(yv, wv)

    print(">> Initialising optimiser and scheduler")
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyperparams["lr"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hyperparams["gamma"])

    self.train_loss = []
    self.validation_loss = []

    print(">> Calculating epoch size")
    epoch_size = min([int(sum(yt==0)/len(self.unique_combinations)), sum(yt==1)])*2 #epoch size is 2*nbkg or 2*nsig, whatever is smallest

    #find indicies for each mass, useful for calculating loss later
    print(">> Finding mass indicies")
    t_idx = []
    v_idx = []
    for mass in self.unique_combinations:
      t_idx.append((Xt[:,-self.n_params:]==mass).sum(axis=1) == self.n_params)
      v_idx.append((Xv[:,-self.n_params:]==mass).sum(axis=1) == self.n_params)

    with tqdm(range(self.hyperparams["max_epochs"])) as t:
      for i_epoch in t:
        self.model.train()
        for batch_X, batch_y, batch_w in tqdm(self.getBatches(Xt, yt, wt, self.hyperparams["batch_size"], shuffle=True, weighted=True, epoch_size=epoch_size), leave=False):
          optimizer.zero_grad()
          loss = self.BCELoss(self.model(batch_X), batch_y, batch_w)
          loss.backward()
          optimizer.step()
        
        self.model.eval()
        with torch.no_grad():
          tl = []
          vl = []

          #calculate loss over different masses
          for i, mass in enumerate(self.unique_combinations):
            s = t_idx[i]
            tl.append(self.getTotLoss(Xt[s], yt[s], wt[s]) * len(Xt[s]))
            s = v_idx[i]
            vl.append(self.getTotLoss(Xv[s], yv[s], wv[s]) * len(Xv[s]))
            # s = t_idx[i]
            # tl.append(self.getTotLoss(Xt, yt, wt, s) * len(Xt[s]))
            # s = v_idx[i]
            # vl.append(self.getTotLoss(Xv, yv, wv, s) * len(Xv[s]))
          
          self.train_loss.append(np.array(tl))
          self.validation_loss.append(np.array(vl))

          t.set_postfix(train_loss=self.train_loss[-1].sum(), validation_loss=self.validation_loss[-1].sum(), gamma=scheduler.get_last_lr()[0])
          
          if self.outdir != None:
            if self.validation_loss[-1].sum() == np.array(self.validation_loss).sum(axis=1).min(): #if best loss is current loss
              self.saveModel()

          if self.shouldSchedulerStep():
            scheduler.step()

          #self.updateLossPlot(train_loss, test_loss, scheduler.get_last_lr()[0])

          if self.shouldEarlyStop():
            break

    if self.outdir != None:
      print("Loading best model")
      self.loadModel()

    self.train_loss = np.array(self.train_loss)
    self.validation_loss = np.array(self.validation_loss)
    self.mass_key = self.unique_combinations

    print("Finished training")

  def predict_proba(self, X, batch_size=8192):
    self.model.eval()
    with torch.no_grad():

      #find unique combinations of parameters (masses)
      unique_combinations = np.unique(X[:,-self.n_params:], axis=0)
      assert len(unique_combinations)==1, print("Expect only one combination of parameters (masses) when evaluating ParamBDT")

      # predictions = []
      # for i_picture in range(0, len(X), batch_size):
      #   batch_X = X[i_picture:i_picture + batch_size]
      #   X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      #   predictions.append(self.model(X_torch).to('cpu').detach().numpy())
      # all_predictions = np.concatenate(predictions)

      X_torch = torch.tensor(X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      all_predictions = self.model(X_torch).to('cpu').detach().numpy()

      return np.concatenate([(1-all_predictions)[:,np.newaxis], all_predictions[:,np.newaxis]], axis=1) #get into format expected by sklearn / xgboost