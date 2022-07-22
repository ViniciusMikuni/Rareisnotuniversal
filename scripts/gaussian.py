import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import utils, models
import argparse

utils.SetStyle()
tf.random.set_seed(1234)

parser = argparse.ArgumentParser()

parser.add_argument('--plot_folder', default='../plots', help='Folder to store plots')
parser.add_argument('--ndim', type=int,default=2, help='Number of dimensions to use oin the toy problem')
parser.add_argument('--max_epoch', type=int,default=500, help='Maximum number of epochs to train')
parser.add_argument('--lr', type=float,default=1e-3, help='Learning rate')

flags = parser.parse_args()

plot_folder=flags.plot_folder
num_dim=flags.ndim
num_epoch = flags.max_epoch

#Generate gaussians samples and respective transformed cdf coordinates
train_b, train_s, train_cdf_b, train_cdf_s, train_tanh_b, train_tanh_s = utils.DataGenerator(num_dim=num_dim)

roc_curves = {}

#Normalizing flow
models.FFJORD(train_b,'checkpoint_Flow_gaus',lr=flags.lr,max_epoch = num_epoch)
models.FFJORD(train_cdf_b,'checkpoint_Flow_cdf',lr=flags.lr,max_epoch = num_epoch)
models.FFJORD(train_tanh_b,'checkpoint_Flow_tanh',lr=flags.lr,max_epoch = num_epoch)

# #Autoencoder training
models.AE(train_b,'checkpoint_AE_gaus',lr=flags.lr,max_epoch = num_epoch)
models.AE(train_cdf_b,'checkpoint_AE_cdf',lr=flags.lr,max_epoch = num_epoch)
models.AE(train_tanh_b,'checkpoint_AE_tanh',lr=flags.lr,max_epoch = num_epoch)

#Weakly supervised training
data_x = np.concatenate([train_s,train_b],0)
data_cdf = np.concatenate([train_cdf_s,train_cdf_b],0)
data_tanh = np.concatenate([train_tanh_s,train_tanh_b],0)

models.CWoLa(data_x,train_b,'checkpoint_CWoLa_gaus',lr=flags.lr,max_epoch = num_epoch)
models.CWoLa(data_cdf,train_cdf_b,'checkpoint_CWoLa_cdf',lr=flags.lr,max_epoch = num_epoch)
models.CWoLa(data_tanh,train_tanh_b,'checkpoint_CWoLa_tanh',lr=flags.lr,max_epoch = num_epoch)

