import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import utils, models
import argparse

utils.SetStyle()
tf.random.set_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
parser.add_argument('--ndim', type=int,default=2, help='Number of dimensions to use oin the toy problem')
flags = parser.parse_args()
plot_folder=flags.plot_folder
num_dim=flags.ndim

val_b, val_s, val_cdf_b, val_cdf_s, val_tanh_b, val_tanh_s = utils.DataGenerator(num_dim=num_dim)


#Let's plot how they look like
feed_dict={
    'signal':val_s[:,0],
    'background':val_b[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='g(x)',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_gauss"))



feed_dict={
    'signal':val_cdf_s[:,0],
    'background':val_cdf_b[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='g(x)',
                           label_loc='upper left',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_cdf"))

feed_dict={
    'signal':val_tanh_s[:,0],
    'background':val_tanh_b[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='g(x)',
                           binning=np.linspace(-1,1,20),
                           ylabel='Normalized events',
                           label_loc='upper left',
                           logy=True,
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_tanh"))

roc_curves = {}

fpr_gaus, tpr_gaus= models.FFJORD([],'checkpoint_Flow_gaus',[val_s,val_b],load=True)
roc_curves['Gaussian NF'] = [fpr_gaus, tpr_gaus]
fpr_cdf, tpr_cdf= models.FFJORD([],'checkpoint_Flow_cdf',[val_cdf_s,val_cdf_b],load=True)
roc_curves['CDF NF'] =[fpr_cdf, tpr_cdf]
fpr_tanh, tpr_tanh= models.FFJORD([],'checkpoint_Flow_tanh',[val_tanh_s,val_tanh_b],load=True)
roc_curves['tanh NF'] =[fpr_tanh, tpr_tanh]

#Autoencoder training
fpr_gaus, tpr_gaus= models.AE([],'checkpoint_AE_gaus',[val_s,val_b],load=True)
roc_curves['Gaussian AE'] = [fpr_gaus, tpr_gaus]
fpr_cdf, tpr_cdf= models.AE([],'checkpoint_AE_cdf',[val_cdf_s,val_cdf_b],load=True)
roc_curves['CDF AE'] =[fpr_cdf, tpr_cdf]
fpr_tanh, tpr_tanh= models.AE([],'checkpoint_AE_tanh',[val_tanh_s,val_tanh_b],load=True)
roc_curves['tanh AE'] = [fpr_tanh, tpr_tanh]



#Weakly supervised training
fpr_gaus, tpr_gaus= models.CWoLa([],[],'checkpoint_CWoLa_gaus',[val_s,val_b],load=True)
roc_curves['Gaussian CWoLa'] =[fpr_gaus, tpr_gaus]
fpr_cdf, tpr_cdf= models.CWoLa([],[],'checkpoint_CWoLa_cdf',[val_cdf_s,val_cdf_b],load=True)
roc_curves['CDF CWoLa'] =[fpr_cdf, tpr_cdf]
fpr_tanh, tpr_tanh= models.CWoLa([],[],'checkpoint_CWoLa_tanh',[val_tanh_s,val_tanh_b],load=True)
roc_curves['tanh CWoLa'] =[fpr_tanh, tpr_tanh]


x = np.linspace(0,1,1000)


fig,gs = utils.SetFig("True positive rate","1 - Fake Rate") 
for plot in roc_curves:
    fpr,tpr = roc_curves[plot]
    plt.plot(tpr,1-fpr,label="{}".format(plot),
             color=utils.colors[plot],linestyle=utils.line_style[plot])
plt.plot(x,1-x,color='black',linestyle='-')

    
plt.ylim([-0.25, 1.1])
plt.legend(frameon=False,fontsize=12,ncol=3)
fig.savefig('{}/{}.pdf'.format(plot_folder,"ROC_gaus"))
