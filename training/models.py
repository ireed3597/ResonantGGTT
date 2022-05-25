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
  def initModel(self, hyperparams):
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


class ParamModel(Model):
  def __init__(self, n_params, n_sig_procs):
    self.n_params = n_params
    self.n_sig_procs = n_sig_procs
    self.initModel()

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

#hyperparams = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 0.01, 'missing': np.nan, 'n_estimators': 200, 'n_jobs': 1, 'nthread': None, 'objective': 'binary:logistic', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': None, 'silent': True, 'subsample': 1, 'eta': 0.3, 'maxDepth': 6}
hyperparams = {'colsample_bytree':0.6, 'eta':0.3, 'maxDepth':6, 'min_child_weight':0.01}

class ParamBDT(ParamModel):
  def initModel(self):
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

class ParamNN(Model):
  def initModel(self):
    # self.model = torch.nn.Sequential(
    #               torch.nn.Linear(len(self.train_features),1),
    #               torch.nn.ELU(),
    #               torch.nn.Flatten(0,1),
    #               torch.nn.Sigmoid()
    #               )

    # self.model = torch.nn.Sequential(
    #               torch.nn.Linear(len(self.train_features),10),
    #               torch.nn.ELU(),
    #               torch.nn.Linear(10,10),
    #               torch.nn.ELU(),
    #               torch.nn.Linear(10,1),
    #               torch.nn.Flatten(0,1),
    #               torch.nn.Sigmoid()
    #             )

    nfeatures = len(self.train_features)
    self.model = torch.nn.Sequential(
                  torch.nn.Linear(nfeatures,int(nfeatures/2)),
                  torch.nn.Dropout(0.1),
                  torch.nn.ELU(),
                  torch.nn.Linear(int(nfeatures/2),int(nfeatures/2)),
                  torch.nn.Dropout(0.1),
                  torch.nn.ELU(),
                  torch.nn.Linear(int(nfeatures/2),1),
                  torch.nn.Flatten(0,1),
                  torch.nn.Sigmoid()
                )

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

  def getTotLoss(self, X, y, w, batch_size):
    losses = []
    for batch_X, batch_y, batch_w in self.getBatches(X, y, w, batch_size):
      loss = self.BCELoss(self.model(batch_X), batch_y, batch_w)
      losses.append(loss.item())
    return sum(losses)

  def getBatches(self, X, y, w, batch_size, shuffle=False):
    if shuffle:
      shuffle_ids = np.random.permutation(len(X))
      X_sh = X[shuffle_ids].copy()
      y_sh = y[shuffle_ids].copy()
      w_sh = w[shuffle_ids].copy()
    else:
      X_sh = X.copy()
      y_sh = y.copy()
      w_sh = w.copy()
    for i_picture in range(0, len(X), batch_size):
      batch_X = X_sh[i_picture:i_picture + batch_size]
      batch_y = y_sh[i_picture:i_picture + batch_size]
      batch_w = w_sh[i_picture:i_picture + batch_size]
    
      X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      y_torch = torch.tensor(batch_y, dtype=torch.float).to(dev)
      w_torch = torch.tensor(batch_w, dtype=torch.float).to(dev)

      yield X_torch, y_torch, w_torch

  def shouldEarlyStop(self, losses, min_epoch=10, grace_epochs=5, tol=0.01):
    """
    Want to stop if seeing no appreciable improvment.
    Check 1. Was the best score more than grace_epochs epochs ago?
          2. If score is improving, has it improved by more than
             tol percent over grace_epochs?
    """

    n_epochs = len(losses)

    if n_epochs < min_epoch:
      return False

    losses = np.array(losses)
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

    if n_epochs - best_loss_epoch > grace_epochs:
      print("Best loss happened a while ago")
      return True

    #check to see if any mass points making good progress
    if n_epochs > grace_epochs:
      any_make_good_improvement = False
      for i in range(len(self.train_masses)):
        best_loss = losses[:,i].min()        
        best_loss_before = losses[:,i][:-grace_epochs].min()
        if (best_loss_before-best_loss)/best_loss > tol:
          any_make_good_improvement = True
          break
      if not any_make_good_improvement:
        print("Not enough improvement")
        return True

    return False

  def shouldSchedulerStep(self, losses):
    n_epochs = len(losses)
    
    losses = np.array(losses)
    slosses = losses.sum(axis=1)
    best_loss = slosses.min()
    best_loss_epoch = np.where(slosses==best_loss)[0][0] + 1

    if ((n_epochs - best_loss_epoch) > 10) and ((n_epochs - self.last_step_epoch) > 10):
      self.last_step_epoch = n_epochs
      return True
    else:
      return False

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

  def train(self, X_train, y_train, X_test, y_test, w_train, w_test, max_epochs=50, batch_size=128, lr=0.001, min_epoch=50, grace_epochs=5, tol=0.01, gamma=0.9, save_location=None):
    self.alreadyPlotting = False
    self.last_step_epoch = 0

    X_train, y_train, w_train = self.inflateBkgWithMasses(X_train, y_train, w_train)
    X_test, y_test, w_test = self.inflateBkgWithMasses(X_test, y_test, w_test)
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()
    w_train = w_train.to_numpy()
    w_test = w_test.to_numpy()

    #X_train, y_train, w_train = self.cullSignal(X_train, y_train, w_train)
    #X_test, y_test, w_test = self.cullSignal(X_test, y_test, w_test)

    nbkg = sum(y_train==0)
    w_train = self.equaliseWeights(X_train, w_train, y_train, norm=nbkg*10)
    w_test = self.equaliseWeights(X_test, w_test, y_test, norm=nbkg*10)
    
    X_train, y_train, w_train = self.smoothBkgWeights(X_train, y_train, w_train, threshold=2)
    #X_test, y_test, w_test = self.smoothBkgWeights(X_test, y_test, w_test, threshold=2)
    print("Minimum weight in training set: ", min(abs(w_train[y_train==0])))
    print("Maximum weight in training set: ", max(abs(w_train[y_train==0])))

    #w_train = self.reweightMass(X_train, y_train, w_train, 0.3, 100)
    #w_test = self.reweightMass(X_test, y_test, w_test, 0.3, 100)

    self.printSampleSummary(X_train, y_train, X_test, y_test, w_train, w_test)

    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
    #optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    train_loss = []
    test_loss = []

    if save_location != None:
      os.makedirs(save_location, exist_ok=True)

    with tqdm(range(max_epochs)) as t:
      for i_epoch in t:
        #X_train, X_test = self.shuffleBkg(X_train, y_train, X_test, y_test)

        for batch_X, batch_y, batch_w in tqdm(self.getBatches(X_train, y_train, w_train, batch_size, shuffle=True), leave=False):
          loss = self.BCELoss(self.model(batch_X), batch_y, batch_w)
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          
        trl = []
        tel = []
        for mass in self.train_masses:
          trl.append(self.getTotLoss(X_train[X_train[:,-1]==mass], y_train[X_train[:,-1]==mass], w_train[X_train[:,-1]==mass], 8192))
          tel.append(self.getTotLoss(X_test[X_test[:,-1]==mass], y_test[X_test[:,-1]==mass], w_test[X_test[:,-1]==mass], 8192))
        train_loss.append(np.array(trl))
        test_loss.append(np.array(tel))

        t.set_postfix(train_loss=train_loss[-1].sum(), test_loss=test_loss[-1].sum(), gamma=scheduler.get_last_lr()[0])
        
        if save_location != None:
          if test_loss[-1].sum() == np.array(test_loss).sum(axis=1).min(): #if best loss is current loss
            # with open("%s/model.pt"%save_location, "wb") as f:
            #   torch.save(self.model, f)
            torch.save(self.model, "%s/model.pt"%save_location)

        if self.shouldSchedulerStep(train_loss):
          scheduler.step()

        #self.updateLossPlot(train_loss, test_loss, scheduler.get_last_lr()[0])

        if self.shouldEarlyStop(test_loss, min_epoch=min_epoch, grace_epochs=grace_epochs, tol=tol):
          break

    if save_location != None:
      print("Loading best model")
      self.model = torch.load("%s/model.pt"%save_location)

    print("Finished training")
      
    return train_loss, test_loss

  def predict(self, X, batch_size=32):
    X = X.to_numpy()
    predictions = []
    for i_picture in range(0, len(X), batch_size):
      batch_X = X[i_picture:i_picture + batch_size]
      X_torch = torch.tensor(batch_X, dtype=torch.float).reshape(-1, X.shape[1]).to(dev)
      predictions.append(self.model(X_torch).to('cpu').detach().numpy())
    return np.concatenate(predictions)
