import numpy as np


def cross_entropy(targets,preds, epsilon=1e-07):
    return - np.sum(targets * np.log(preds + epsilon))

def kl_divergence(target_distribution, predicted_distribution):
    ce = cross_entropy(target_distribution, predicted_distribution)
    entropy = cross_entropy(target_distribution, target_distribution)
    return ce - entropy


    