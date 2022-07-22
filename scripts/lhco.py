import numpy as np
import matplotlib.pyplot as plt
import sys,os
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import utils, models
import argparse
import pandas as pd


utils.SetStyle()

parser = argparse.ArgumentParser()


parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/LHCO/', help='Folder to store plots')
parser.add_argument('--max_epoch', type=int,default=500, help='Maximum number of epochs to train')
parser.add_argument('--lr', type=float,default=1e-3, help='Learning rate')

flags = parser.parse_args()
num_epoch = flags.max_epoch


sig = pd.read_hdf(os.path.join(flags.data_folder,"events_anomalydetection_DelphesPythia8_v2_Wprime_features.h5")).to_numpy()
bkg = pd.read_hdf(os.path.join(flags.data_folder,"events_anomalydetection_DelphesPythia8_v2_qcd_features.h5")).to_numpy()

tau2_b = bkg[:125000,5]
tau2_s = sig[:,5]
eps=1e-3
tau1_b = bkg[:125000,4]
tau1_s = sig[:,4]

tau1tau2_b = np.stack([tau1_b,tau2_b],-1)
tau1tau2_s = np.stack([tau1_s,tau2_s],-1)


tau1tau12_b = np.stack([tau1_b,tau2_b/(eps+tau1_b)],-1)
tau1tau12_s = np.stack([tau1_s,tau2_s/(eps+tau1_s)],-1)


train_tau1tau2_b,_=train_test_split(tau1tau2_b, test_size=0.2,shuffle=False)
train_tau1tau12_b,_=train_test_split(tau1tau12_b, test_size=0.2,shuffle=False)

train_tau1tau2_s,_=train_test_split(tau1tau2_s, test_size=0.2,shuffle=False)
train_tau1tau12_s,_=train_test_split(tau1tau12_s, test_size=0.2,shuffle=False)


# #Normalizing flow
# models.FFJORD(train_tau1tau2_b,'checkpoint_Flow_tau1tau2',lr=flags.lr,max_epoch = num_epoch)
# models.FFJORD(train_tau1tau12_b,'checkpoint_Flow_tau1tau12',lr=flags.lr,max_epoch = num_epoch)
# roc_curves[r'$(\tau_1,\tau_2/\tau_1)$ NF'] =[fpr_tau1tau12, tpr_tau1tau12]



# #Autoencoder training
# models.AE(train_tau1tau2_b,'checkpoint_AE_tau1tau2',lr=flags.lr,max_epoch = num_epoch)
# models.AE(train_tau1tau12_b,'checkpoint_AE_tau1tau12',lr=flags.lr,max_epoch = num_epoch)

#Weakly supervised training
data_tau1tau2 = np.concatenate([train_tau1tau2_s,train_tau1tau2_b],0)
data_tau1tau12 = np.concatenate([train_tau1tau12_s,train_tau1tau12_b],0)

models.CWoLa(data_tau1tau2,train_tau1tau2_b,'checkpoint_CWoLa_tau1tau2',lr=flags.lr,max_epoch = num_epoch)
models.CWoLa(data_tau1tau12,train_tau1tau12_b,'checkpoint_CWoLa_tau1tau12',lr=flags.lr,max_epoch = num_epoch)
