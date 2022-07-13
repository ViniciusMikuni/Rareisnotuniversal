import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import utils
import argparse

utils.SetStyle()

parser = argparse.ArgumentParser()


parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
parser.add_argument('--ndim', type=int,default=10, help='Number of dimensions to use oin the toy problem')
parser.add_argument('--max_epoch', type=int,default=500, help='Maximum number of epochs to train')
parser.add_argument('--lr', type=float,default=1e-3, help='Learning rate')

flags = parser.parse_args()

plot_folder=flags.plot_folder
num_dim=flags.ndim
num_epoch = flags.max_epoch

#Generate gaussians samples and respective transformed cdf coordinates
x_b, x_s, cdf_s, cdf_b = utils.DataGenerator(num_dim=num_dim)


#Let's plot how they look like
feed_dict={
    'signal':x_s[:,0],
    'background':x_b[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='feature 0',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_feature0"))



feed_dict={
    'signal':cdf_s[:,0],
    'background':cdf_b[:,0],
}

fig,ax = utils.HistRoutine(feed_dict,plot_ratio=False,
                           xlabel='feature 0',
                           ylabel='Normalized events',
                           reference_name='signal')


fig.savefig('{}/{}.pdf'.format(plot_folder,"Hist1D_feature0_cdf"))


opt = tf.keras.optimizers.Adam(learning_rate=flags.lr)
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]

roc_curves = {}


#Autoencoder training

def Unsup(x_s,x_b,checkpoint_name):
    '''
    Inputs:
    x_s: Signal events. Only used to calculate the fpr and tpr
    x_b: Background events used for training
    checkpoint_name: Name of the folder to store trained models
    Outputs:
    fpr: false positive rate calculated using the reconstruction loss of the autoencoder
    tpr: true positive rate calculated using the reconstruction loss of the autoencoder
    '''
    K.clear_session()
    inputs,outputs = utils.AE(num_dim,NLAYERS=3,LAYERSIZE=[20,10,5],ENCODESIZE=3)
    model = Model(inputs, outputs)
    model.compile(loss="mse", optimizer=opt, metrics=['accuracy'])
    checkpoint = ModelCheckpoint('../checkpoint/{}.hdf5'.format(checkpoint_name),
                                 save_best_only=True,mode='auto',period=1,save_weights_only=True)

    hist_ae = model.fit(x_b,x_b,
                        epochs=num_epoch, 
                        callbacks=callbacks+[checkpoint]
,
                        validation_split=0.2,
                        batch_size=1024)
    
    AE_b = model.predict(x_b,batch_size=1000)
    mse_AE_b = np.mean(np.square(AE_b - x_b),-1)
    AE_s = model.predict(x_s,batch_size=1000)
    mse_AE_s = np.mean(np.square(AE_s - x_s),-1)
    mse = np.concatenate([mse_AE_b,mse_AE_s],0)
    label = np.concatenate([np.zeros(mse_AE_b.shape[0]),np.ones(mse_AE_s.shape[0])],0)

    fpr, tpr, _ = roc_curve(label,mse, pos_label=1)    
    print("AE AUC: {}".format(auc(fpr, tpr)))

    return fpr,tpr

fpr_gaus, tpr_gaus= Unsup(x_s,x_b,'checkpoint_AE_gaus')
roc_curves['Gaussian AE'] = [fpr_gaus, tpr_gaus]
fpr_cdf, tpr_cdf= Unsup(cdf_s,cdf_b,'checkpoint_AE_cdf')
roc_curves['CDF AE'] =[fpr_cdf, tpr_cdf]

#Weakly supervised training

def CWoLa(data,label,real_label,checkpoint_name):
    '''
    Inputs: 
    data: full dataset containing both signal and background events
    label: Weakly supervised label to use to define the mixed regions
    real_label: Truth label to identify signal and background events
    checkpoint_name: Name of the folder to store trained models

    Outputs:
    fpr: false positive rate calculated using the ratio ps/pb
    tpr: true positive rate calculated using the ratio ps/pb
    '''
    K.clear_session()
    inputs,outputs = utils.Classifier(num_dim-1,NLAYERS=3,LAYERSIZE=[20,10,5])    
    model = Model(inputs, outputs)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=['accuracy'])
    checkpoint = ModelCheckpoint('../checkpoint/{}.hdf5'.format(checkpoint_name),
                                 save_best_only=True,mode='auto',period=1,save_weights_only=True)
    hist_ae = model.fit(data,label,
                        epochs=num_epoch, 
                        callbacks=callbacks+[checkpoint],
                        validation_split=0.2,
                        batch_size=1024)

    
    pred = model.predict(data,batch_size=1000)
    pred = pred/(1-pred)
    fpr, tpr, _ = roc_curve(real_label,pred, pos_label=1)    
    print("CWoLa AUC: {}".format(auc(fpr, tpr)))

    return fpr,tpr

real_label = np.concatenate([np.ones(x_s.shape[0]),np.zeros(x_b.shape[0])],0)
data_x = np.concatenate([x_s,x_b])
data_cdf = np.concatenate([cdf_s,cdf_b])
weak_label = data_x[:,0]>0.5 #threshold used to create mixed regions: first dim is used to create regions while the others are used for training



fpr_gaus, tpr_gaus= CWoLa(data_x[:,1:],weak_label,real_label,'checkpoint_CWoLa_gaus')
roc_curves['Gaussian CWoLa'] =[fpr_gaus, tpr_gaus]


fpr_cdf, tpr_cdf= CWoLa(data_cdf[:,1:],weak_label,real_label,'checkpoint_CWoLa_cdf')
roc_curves['CDF CWoLa'] =[fpr_cdf, tpr_cdf]

fig,gs = utils.SetFig("True positive rate","1 - Fake Rate") 
for plot in roc_curves:
    fpr,tpr = roc_curves[plot]
    plt.plot(tpr,1-fpr,label="{} ({:.2f})".format(plot,auc(tpr,1-fpr)))
    
plt.legend(frameon=False,fontsize=14)
fig.savefig('{}/{}.pdf'.format(plot_folder,"ROC"))
