import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import utils, models
import argparse

utils.SetStyle()

parser = argparse.ArgumentParser()


parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
parser.add_argument('--ndim', type=int,default=3, help='Number of dimensions to use oin the toy problem')
parser.add_argument('--max_epoch', type=int,default=500, help='Maximum number of epochs to train')
parser.add_argument('--lr', type=float,default=1e-3, help='Learning rate')

flags = parser.parse_args()

plot_folder=flags.plot_folder
num_dim=flags.ndim
num_epoch = flags.max_epoch

#Generate gaussians samples and respective transformed cdf coordinates
train_b, train_s, train_cdf_b, train_cdf_s = utils.DataGenerator(num_dim=num_dim)
val_b, val_s, val_cdf_b, val_cdf_s = utils.DataGenerator(num_dim=num_dim)


#Let's plot how they look like
feed_dict={
    'signal':train_s[:,0],
    'background':train_b[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='feature 0',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_feature0"))



feed_dict={
    'signal':train_cdf_s[:,0],
    'background':train_cdf_b[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='feature 0',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_feature0_cdf"))



roc_curves = {}

#Normalizing flow
fpr_gaus, tpr_gaus= models.FFJORD(train_b,[val_s,val_b],'checkpoint_Flow_gaus',lr=flags.lr,max_epoch = num_epoch)
roc_curves['Gaussian NF'] = [fpr_gaus, tpr_gaus]
fpr_cdf, tpr_cdf= models.FFJORD(train_cdf_b,[val_cdf_s,val_cdf_b],'checkpoint_Flow_cdf',lr=flags.lr,max_epoch = num_epoch)
roc_curves['CDF NF'] =[fpr_cdf, tpr_cdf]



#Autoencoder training
fpr_gaus, tpr_gaus= models.AE(train_b,[val_s,val_b],'checkpoint_AE_gaus',lr=flags.lr,max_epoch = num_epoch)
roc_curves['Gaussian AE'] = [fpr_gaus, tpr_gaus]
fpr_cdf, tpr_cdf= models.AE(train_cdf_b,[val_cdf_s,val_cdf_b],'checkpoint_AE_cdf',lr=flags.lr,max_epoch = num_epoch)
roc_curves['CDF AE'] =[fpr_cdf, tpr_cdf]



#Weakly supervised training
data_x = np.concatenate([train_s,train_b],0)
data_cdf = np.concatenate([train_cdf_s,train_cdf_b],0)

fpr_gaus, tpr_gaus= models.CWoLa(data_x,train_b,[val_s,val_b],'checkpoint_CWoLa_gaus',lr=flags.lr,max_epoch = num_epoch)
roc_curves['Gaussian CWoLa'] =[fpr_gaus, tpr_gaus]

fpr_cdf, tpr_cdf= models.CWoLa(data_cdf,val_cdf_b,[val_cdf_s,val_cdf_b],'checkpoint_CWoLa_cdf',lr=flags.lr,max_epoch = num_epoch)
roc_curves['CDF CWoLa'] =[fpr_cdf, tpr_cdf]





fig,gs = utils.SetFig("True positive rate","1 - Fake Rate") 
for plot in roc_curves:
    fpr,tpr = roc_curves[plot]
    plt.plot(tpr,1-fpr,label="{} ({:.2f})".format(plot,auc(tpr,1-fpr)),
             color=utils.colors[plot],linestyle=utils.line_style[plot])
    
plt.legend(frameon=False,fontsize=14)
fig.savefig('{}/{}.pdf'.format(plot_folder,"ROC_gaus"))
