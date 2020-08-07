####################
# General Imports  #
####################
import torch
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch import nn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import matplotlib.pyplot as plt # Plot Graphs
from torchsummary import summary
import numpy as np
import pickle
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import datetime
import cv2
import os
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from skimage.segmentation import mark_boundaries
from lime import lime_image


def forward_step(dataloader, net, flags, loss_fun):
    """
    Makes the step of the training phase

    Arguments
    ---------
    dataloader, network, flags, and loss function

    """
    device_name, device = check_cuda()
    total_samples, correct_samples = 0, 0
    loss_arr = []
    # Makes the step
    for labels, aug_inputs in dataloader:
        labels = labels.cuda()
        for inputs in aug_inputs:
            inputs = inputs.cuda()
            if flags.Rgb:
                inputs = inputs.repeat(1, 3, 1, 1)
            # Calculate loss
            output = net(inputs)
            labels = torch.tensor(labels, dtype=torch.long, device=device).cuda()
            loss = loss_fun(output, labels).item()
            loss_arr.append(loss)
            predicts = torch.max(output, 1)[1]
            correct_samples += (predicts == labels).sum().item()
            total_samples += len(labels)


def check_cuda():
    """
    Check if the cuda exists in the computer

    Arguments
    ---------
    dataloader, network, flags, and loss function

    """
    device_name = "cuda:0:" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    if torch.cuda.device_count() > 0:
        print("Cuda is connected.\n")
        torch.cuda.empty_cache()
    else:
        print("Error: Cannot use cuda.\n")
        sys.exit(1)
    return device_name, device


def k_vold_index_divide(idx, idx_len, n_splits, current_group):
    """
    Makes the k fold cross validation

    Arguments
    ---------
    idx - the idx dedicated to tranining phase
    idx_len - number of images in the training phase
    n_splits - number of folds
    current group - the curent fold

    Outputs
    ---------
    train idx - the idx for tranining
    valid_idx - the idx for validation
    current group - the curent fold
    """

    group_size = int(idx_len/n_splits)
    if current_group == n_splits-1:
        valid_idx = idx[current_group*group_size:]
        train_idx = idx[:current_group*group_size]
        current_group = 0
    else:
        valid_idx = idx[current_group * group_size:(current_group + 1) * group_size]
        train_idx = np.concatenate((idx[:current_group * group_size],
                                    idx[(current_group + 1) * group_size:]))
        current_group += 1
    return train_idx, valid_idx, current_group


def plot_confusion_matrix(flags, cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    if flags.SaveFig:
        fig_path = os.path.join(flags.model_dir,
                                  flags.check_name[:-3] + '_cm.png')
        print ("Save fig:" + fig_path)
        plt.savefig(fig_path)


def evaluate_metrics(actual, predicted, target_names, flags):
    """
    Makes the evaluation

    Arguments
    ---------
    actual - the ground thruth
    predicted - the network output
    target_names - the classes
    flags

    Outputs
    ---------
    cm - confusion matrix
    classification report
    """

    cm = confusion_matrix(actual, predicted)
    report = classification_report(actual, predicted)
    print(report)
    plot_confusion_matrix(flags, cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True)
    return cm, report


def remove_file (path):
    """
    Delete file if exist

    Arguments
    ---------
    path to the file
    """
    if os.path.exists(path):
        print("Delete: " + path)
        os.remove(path)
    else:
        print("The file does not exist")


def plot_loss_accuracy(flags, results):
    """
    Plot the loss and the accuracy

    Arguments
    ---------
    flags
    result class

    """

    for train, val, cat in [[results.train_loss, results.valid_loss, 'loss'],
                            [results.train_accuracy, results.valid_accuracy, 'accuracy']]:
        # Get num of epochs
        epochs = results.epoch  #np.arange(0, len(train)) + 1
        # Plot loss history
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, train, "r--")
        plt.plot(epochs, val, "b-")
        plt.legend(["Train " + cat, "Test " + cat])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid()

        if flags.SaveFig:
            fig_path = os.path.join(flags.model_dir,
                                    flags.check_name[:-3] + '_' + cat + '.png')
            print("Save fig:" + fig_path)
            plt.savefig(fig_path)
        else:
            plt.show()


def smart_vstack(array, vector):
    """
    expand vstack to one element

    Arguments
    ---------
    array
    vector

    Outputs
    ---------
    vstack array

    """
    if array == []:
        return vector
    else:
        return np.vstack((array,vector))


def smart_append(array, vector):
    """
    expand append to one element

    Arguments
    ---------
    array
    vector

    Outputs
    ---------
    append array

    """
    if array == []:
        return vector
    else:
        return np.append((array,vector))


def smart_concatenate(array, vector):
    """
    expand concatenate to one element

    Arguments
    ---------
    array
    vector

    Outputs
    ---------
    concatenate array

    """
    if array == []:
        return vector
    else:
        return np.concatenate((array,vector))


def scale_to_01_range(x):
    """
    scale and move the coordinates so they fit [0; 1] range

    Arguments
    ---------
    x - intput


    Outputs
    ---------
    normalized x

    """
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range