# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:52:40 2019

source: https://github.com/chasingbob/keras-visuals/blob/master/visual_callbacks.py
"""

from keras.callbacks import Callback
import matplotlib.pyplot as plt    
import matplotlib.patches as mpatches  
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np


class AccLossPlotter(Callback):
    """Plot training Accuracy and Loss values on a Matplotlib graph. 
    The graph is updated by the 'on_epoch_end' event of the Keras Callback class
    # Arguments
        graphs: list with some or all of ('acc', 'loss')
        save_graph: Save graph as an image on Keras Callback 'on_train_end' event 
    """

    def __init__(self, graphs=['acc', 'loss'], save_graph=False):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_graph


    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0
        plt.ion()
        plt.show()


    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        epochs = [x for x in range(self.epoch_count)]

        count_subplots = 0
        
        if 'acc' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Accuracy')
            #plt.axis([0,100,0,1])
            plt.plot(epochs, self.val_acc, color='r')
            plt.plot(epochs, self.acc, color='b')
            plt.ylabel('accuracy')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if 'loss' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Loss')
            #plt.axis([0,100,0,5])
            plt.plot(epochs, self.val_loss, color='r')
            plt.plot(epochs, self.loss, color='b')
            plt.ylabel('loss')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)
        
        plt.draw()
        plt.pause(0.001)

    def on_train_end(self, logs={}):
        if self.save_graph:
            plt.savefig('training_acc_loss.png')

class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch
    # Arguments
        X_val: The input values 
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title
    """
    def __init__(self, X_val, Y_val, classes, normalize=False, cmap=plt.cm.Blues):
        self.X_val = X_val
        self.Y_val = Y_val
        self.classes = classes
        self.normalize = normalize
        self.cmap = cmap
        plt.ion()
        #plt.show()
        #plt.figure()

    def on_train_begin(self, logs={}):
        pass

    
    def on_epoch_end(self, epoch, logs={}):    
        plt.clf()
        pred = self.model.predict(self.X_val)
        max_pred = np.argmax(pred, axis=1)
        max_y = np.argmax(self.Y_val, axis=1)
        cnf_mat = confusion_matrix(max_y, max_pred)
        if self.normalize:
            cnf_mat = np.round(cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis],2)
        title = 'Confusion Matrix : Epoch'+str(epoch+1)
        fig, ax = plt.subplots(figsize = (7, 7))
        im = ax.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)
        
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(    xticks      = np.arange(cnf_mat.shape[1]),
               yticks      = np.arange(cnf_mat.shape[0]),
               xticklabels = self.classes, 
               yticklabels = self.classes)
        ax.set_title( title,fontdict={'fontsize': 12, 'fontweight': 'heavy'})
        ax.set_ylabel( 'True label',fontdict={'fontsize': 12, 'fontweight': 'book'})
        ax.set_xlabel( 'Predicted label',fontdict={'fontsize': 12, 'fontweight': 'book'})
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45)
        thresh = cnf_mat.max() / 2.
        for i in range(cnf_mat.shape[0]):
            for j in range(cnf_mat.shape[1]):
                ax.text(j, i, cnf_mat[i, j],
                        ha="center", va="center",
                        color="white" if cnf_mat[i, j] > thresh else "black")
        ax.set_xticks(np.arange(0, len(self.classes), 0.5), minor=True)
        ax.set_yticks(np.arange(0, len(self.classes), 0.5), minor=True)
        ax.grid(False)
        
        ax.set_ylim(bottom = cnf_mat.shape[0] - 0.5, top = -0.5)
        fig.tight_layout()
        plt.show()
        plt.pause(0.001)
    
		#fig.savefig('/home/alexandre/KeeLab/keelab/Emmanuelle_SBR/Resultats/Confusion Matrix : Epoch'+str(epoch)+'.png');