import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, mean_absolute_error, mean_squared_error
from .cliff_util import ActivityCliffs


def prc_auc(targets, preds):
    """
    Computes the area under the precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def compute_cls_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true).reshape(y_pred.shape)

    is_valid = y_true ** 2 > 0
    roc_list = []
    for i in range(y_true.shape[1]):
        valid, label, pred = is_valid[:, i], y_true[:, i], y_pred[:, i]
        label = (label[valid] + 1) / 2
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(np.unique(label)) == 2:
            roc_list.append(roc_auc_score(label, pred[valid]))

    roc_auc = np.mean(roc_list)
    # print('Valid ratio: %s' % (np.mean(is_valid)))
    if len(roc_list) == 0:
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc


def compute_cliff_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    roc_list = []
    for i in range(y_true.shape[1]):
        label, pred = y_true[:, i], y_pred[:, i]
        # AUC is only defined when there is at least one positive pretrain_data.
        if len(np.unique(label)) == 2:
            roc_list.append(roc_auc_score(label, pred))

    roc_auc = np.mean(roc_list)
    # print('Valid ratio: %s' % (np.mean(is_valid)))
    if len(roc_list) == 0:
        raise RuntimeError("No positively labeled pretrain_data available. Cannot compute ROC-AUC.")
    return roc_auc


def compute_reg_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    mae_list = []
    rmse_list = []
    for i in range(y_true.shape[1]):
        label, pred = y_true[:, i], y_pred[:, i]
        mae = mean_absolute_error(label, pred)
        rmse = np.sqrt(mean_squared_error(label, pred, squared=True))
        mae_list.append(mae)
        rmse_list.append(rmse)

    mae, rmse = np.mean(mae_list), np.mean(rmse_list)
    return mae, rmse


def calc_rmse(true, pred):
    """ Calculates the Root Mean Square Error

    Args:
        true: (1d array-like shape) true test values (float)
        pred: (1d array-like shape) predicted test values (float)

    Returns: (float) rmse
    """
    # Convert to 1-D numpy array if it's not
    if type(pred) is not np.array:
        pred = np.array(pred)
    if type(true) is not np.array:
        true = np.array(true)

    return np.sqrt(np.mean(np.square(true - pred)))


def calc_cliff_rmse(y_test_pred, y_test, cliff_mols_test=None, smiles_test=None,
                    y_train=None, smiles_train=None, **kwargs):
    """ Calculate the RMSE of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :param smiles_test: (lst) list of SMILES strings of the test molecules
    :param y_train: (lst/array) train labels
    :param smiles_train: (lst) list of SMILES strings of the train molecules
    :param kwargs: arguments for ActivityCliffs()
    :return: float RMSE on activity cliff compounds
    """

    # Check if we can compute activity cliffs when pre-computed ones are not provided.
    if cliff_mols_test is None:
        if smiles_test is None or y_train is None or smiles_train is None:
            raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
                             'to compute activity cliffs')

    # Convert to numpy array if it is none
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    if cliff_mols_test is None:
        y_train = np.array(y_train) if type(y_train) is not np.array else y_train
        # Calculate cliffs and
        cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
        cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
        # Take only the test cliffs
        cliff_mols_test = cliff_mols[len(smiles_train):]

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)
