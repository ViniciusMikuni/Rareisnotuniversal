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


parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/LHCO/', help='Folder to store plots')
parser.add_argument('--max_epoch', type=int,default=500, help='Maximum number of epochs to train')
parser.add_argument('--lr', type=float,default=1e-3, help='Learning rate')

flags = parser.parse_args()
plot_folder=flags.plot_folder
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


_,val_tau1tau2_b=train_test_split(tau1tau2_b, test_size=0.2,shuffle=False)
_,val_tau1tau12_b=train_test_split(tau1tau12_b, test_size=0.2,shuffle=False)

_,val_tau1tau2_s=train_test_split(tau1tau2_s, test_size=0.2,shuffle=False)
_,val_tau1tau12_s=train_test_split(tau1tau12_s, test_size=0.2,shuffle=False)


#Let's plot how they look like
feed_dict={
    'signal':tau1_s,
    'background':tau1_b,
}


fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='tau1',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_tau1"))


feed_dict={
    'signal':tau2_s/(eps+tau1_s),
    'background':tau2_b/(eps+tau1_b),
}


fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='tau12',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_tau12"))


feed_dict={
    'signal':tau2_s,
    'background':tau2_b,
}


fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='tau2',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_tau2"))

roc_curves = {}


#Normalizing flow
fpr_tau1tau2, tpr_tau1tau2= models.FFJORD([],'checkpoint_Flow_tau1tau2',[val_tau1tau2_s,val_tau1tau2_b],load=True)
roc_curves[r'$(\tau_1,\tau_2)$ NF'] = [fpr_tau1tau2, tpr_tau1tau2]
fpr_tau1tau12, tpr_tau1tau12= models.FFJORD([],'checkpoint_Flow_tau1tau12',[val_tau1tau12_s,val_tau1tau12_b],load=True)
roc_curves[r'$(\tau_1,\tau_2/\tau_1)$ NF'] =[fpr_tau1tau12, tpr_tau1tau12]



#Autoencoder training
fpr_tau1tau2, tpr_tau1tau2= models.AE([],'checkpoint_AE_tau1tau2',[val_tau1tau2_s,val_tau1tau2_b],load=True)
roc_curves[r'$(\tau_1,\tau_2)$ AE'] = [fpr_tau1tau2, tpr_tau1tau2]
fpr_tau1tau12, tpr_tau1tau12= models.AE([],'checkpoint_AE_tau1tau12',[val_tau1tau12_s,val_tau1tau12_b],load=True)
roc_curves[r'$(\tau_1,\tau_2/\tau_1)$ AE'] =[fpr_tau1tau12, tpr_tau1tau12]



#Weakly supervised training

fpr_tau1tau2, tpr_tau1tau2= models.CWoLa([],[],'checkpoint_CWoLa_tau1tau2',[val_tau1tau2_s,val_tau1tau2_b],load=True)
roc_curves[r'$(\tau_1,\tau_2)$ CWoLa'] =[fpr_tau1tau2, tpr_tau1tau2]

fpr_tau1tau12, tpr_tau1tau12= models.CWoLa([],[],'checkpoint_CWoLa_tau1tau12',[val_tau1tau12_s,val_tau1tau12_b],load=True)
roc_curves[r'$(\tau_1,\tau_2/\tau_1)$ CWoLa'] =[fpr_tau1tau12, tpr_tau1tau12]


x = np.linspace(0,1,1000)
fig,gs = utils.SetFig("True positive rate","1 - Fake Rate") 
for plot in roc_curves:
    fpr,tpr = roc_curves[plot]
    plt.plot(tpr,1-fpr,label="{} ({:.2f})".format(plot,auc(tpr,1-fpr)),
             color=utils.colors[plot],linestyle=utils.line_style[plot])
plt.plot(x,1-x,color='black',linestyle='-')
plt.legend(frameon=False,fontsize=14,ncol=1)
fig.savefig('{}/{}.pdf'.format(plot_folder,"ROC_lhco"))

