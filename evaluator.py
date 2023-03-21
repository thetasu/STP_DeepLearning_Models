import math
import numpy as np
import scipy.stats as sps
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error,f1_score,precision_score,recall_score,roc_auc_score


def evaluate(prediction, ground_truth, hinge=False, reg=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    if reg:
        performance['mse'] = mean_squared_error(np.squeeze(ground_truth), np.squeeze(prediction))
        return performance

    if hinge:
        pred = (np.sign(prediction) + 1) / 2
        for ind, p in enumerate(pred):
            v = p[0]
            if abs(p[0] - 0.5) < 1e-8 or np.isnan(p[0]):
                pred[ind][0] = 0
    else:
        pred = np.round(prediction)
    try:
        performance['acc'] = accuracy_score(ground_truth, pred)
    except Exception:
        np.savetxt('prediction', pred, delimiter=',')
        exit(0)
    # print('ground_truth.shape:{}'.format(pred.shape))
    performance['mcc'] = matthews_corrcoef(ground_truth, pred)
    performance['f1'] = f1_score(ground_truth, pred)
    performance['pre'] = precision_score(ground_truth, pred)
    performance['rec'] = recall_score(ground_truth, pred)
    performance['auc'] = roc_auc_score(ground_truth, pred)
    return performance


def compare(current_performance, origin_performance):
    is_better = {}
    for metric_name in origin_performance.keys():
        if metric_name == 'mse':
            if current_performance[metric_name] < \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
        else:
            if current_performance[metric_name] > \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
    return is_better
