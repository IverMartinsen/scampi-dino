import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def compute_recall_at_k(labels, dists, k='k'):
    prec_at_k = np.zeros(len(labels))
    rec_at_k = np.zeros(len(labels))

    for i in range(len(labels)):
        if k == 'k':
            _k = np.sum(labels == labels[i]) - 1
        else:
            _k = k
        # get the indices of the k nearest neighbors
        idx = np.argsort(dists[i])[:_k]
        # get the labels of the k nearest neighbors
        nn_labels = labels[idx]
        # count the number of relevant retrieved samples
        n_relevant_retrieved = np.sum(nn_labels == labels[i])
        # count the number of relevant samples
        n_relevant = np.sum(labels == labels[i]) - 1
        # compute the precision at k
        prec_at_k[i] =  n_relevant_retrieved / _k
        # compute the recall at k
        rec_at_k[i] = n_relevant_retrieved / n_relevant
    return prec_at_k, rec_at_k


def compute_recall_at_k_from_ensemble(labels, dists, k='k'):
    prec_at_k = np.zeros(len(labels))
    rec_at_k = np.zeros(len(labels))

    for i in range(len(labels)):
        if k == 'k':
            _k = np.sum(labels == labels[i]) - 1
        else:
            _k = k
        
        # get the indices of the k nearest neighbors
        idx0 = np.argsort(dists[0][i])[:_k]
        idx1 = np.argsort(dists[1][i])[:_k]
        #idx2 = np.argsort(dists[2][i])[:_k]
        
        idx = np.stack([idx0, idx1], axis=1)
        # find the most common index
        idx = np.array([np.argmax(np.bincount(i)) for i in idx])
        
        # get the labels of the k nearest neighbors
        nn_labels = labels[idx]
        # count the number of relevant retrieved samples
        n_relevant_retrieved = np.sum(nn_labels == labels[i])
        # count the number of relevant samples
        n_relevant = np.sum(labels == labels[i]) - 1
        # compute the precision at k
        prec_at_k[i] =  n_relevant_retrieved / _k
        # compute the recall at k
        rec_at_k[i] = n_relevant_retrieved / n_relevant
    return prec_at_k, rec_at_k


def plot_precision_recall_curve(x, y, filename):
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    x = [0] + list(x) + [1]
    y = [1] + list(y) + [0]
    area = auc(x, y)
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label=f"Area under the curve: {area:.2f}", marker="o", linestyle="--", linewidth=2)
    plt.xlabel("Recall", fontsize=15)
    plt.ylabel("Precision", fontsize=15)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def retrieve_filenames(query, labels, filenames, dists):
    filenames = np.array(filenames)
    
    i = np.where(filenames == query)[0][0]
    
    _k = np.sum(labels == labels[i]) - 1
    # get the indices of the k nearest neighbors
    idx = np.argsort(dists[i])[:_k]
    # get the labels of the k nearest neighbors
    nn_labels = labels[idx]
    # get the filenames of the k nearest neighbors
    retrieved_filenames  = [filenames[j] for j in idx]
    return retrieved_filenames
