# 从paddlepaddle中的pahelix中复制
# 有很多种数据划分方式


import random
import numpy as np
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

__all__ = [
    'RandomSplitter',
    'IndexSplitter',
    'ScaffoldSplitter',
    'RandomScaffoldSplitter',
]


def create_splitter(split_type):
    """Return a splitter according to the ``split_type``"""
    if split_type == 'random':
        splitter = RandomSplitter()
    elif split_type == 'index':
        splitter = IndexSplitter()
    elif split_type == 'scaffold':
        splitter = ScaffoldSplitter()
    elif split_type == 'random_scaffold':
        splitter = RandomScaffoldSplitter()
    else:
        raise ValueError('%s not supported' % split_type)
    return splitter


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles: smiles sequence
        include_chirality: Default=False

    Return:
        the scaffold of the given smiles.
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold


class Splitter(object):
    """
    The abstract class of splitters which split up dataset into train/valid/test
    subsets.
    """

    def __init__(self):
        super(Splitter, self).__init__()


class RandomSplitter(Splitter):
    """
    Random splitter.
    """

    def __init__(self):
        super(RandomSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None,
              seed=9):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)
        if isinstance(dataset, list):
            train_dataset = np.array(dataset)[indices[:train_cutoff]].tolist()
            valid_dataset = np.array(dataset)[indices[train_cutoff:valid_cutoff]].tolist()
            test_dataset = np.array(dataset)[indices[valid_cutoff:]].tolist()
        else:
            train_dataset = dataset[indices[:train_cutoff]]
            valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
            test_dataset = dataset[indices[valid_cutoff:]]

        return train_dataset, valid_dataset, test_dataset


class IndexSplitter(Splitter):
    """
    Split daatasets that has already been orderd. The first `frac_train` proportion
    is used for train set, the next `frac_valid` for valid set and the final `frac_test`
    for test set.
    """

    def __init__(self):
        super(IndexSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        indices = list(range(N))
        train_cutoff = int(frac_train * N)
        valid_cutoff = int((frac_train + frac_valid) * N)

        train_dataset = dataset[indices[:train_cutoff]]
        valid_dataset = dataset[indices[train_cutoff:valid_cutoff]]
        test_dataset = dataset[indices[valid_cutoff:]]
        return train_dataset, valid_dataset, test_dataset


class ScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py

    Split dataset by Bemis-Murcko scaffolds
    """

    def __init__(self):
        super(ScaffoldSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the
                scaffold.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        # create dict of the form {scaffold_i: [idx1, idx....]}
        all_scaffolds = {}

        for i, smiles in enumerate(dataset.smiles_list):
            scaffold = generate_scaffold(smiles, include_chirality=False)
            if scaffold not in all_scaffolds:
                all_scaffolds[scaffold] = [i]
            else:
                all_scaffolds[scaffold].append(i)

        # sort from largest to smallest sets
        all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
        all_scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        # get train, valid test indices
        train_cutoff = frac_train * N
        valid_cutoff = (frac_train + frac_valid) * N
        train_idx, valid_idx, test_idx = [], [], []
        for scaffold_set in all_scaffold_sets:
            if len(train_idx) + len(scaffold_set) > train_cutoff:
                if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                    test_idx.extend(scaffold_set)
                else:
                    valid_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        assert len(set(train_idx).intersection(set(valid_idx))) == 0
        assert len(set(test_idx).intersection(set(valid_idx))) == 0

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset


class RandomScaffoldSplitter(Splitter):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/chainer_chemistry/dataset/splitters/scaffold_splitter.py

    Split dataset by Bemis-Murcko scaffolds
    """

    def __init__(self):
        super(RandomScaffoldSplitter, self).__init__()

    def split(self,
              dataset,
              frac_train=None,
              frac_valid=None,
              frac_test=None,
              seed=9):
        """
        Args:
            dataset(InMemoryDataset): the dataset to split. Make sure each element in
                the dataset has key "smiles" which will be used to calculate the
                scaffold.
            frac_train(float): the fraction of pretrain_data to be used for the train split.
            frac_valid(float): the fraction of pretrain_data to be used for the valid split.
            frac_test(float): the fraction of pretrain_data to be used for the test split.
            seed(int|None): the random seed.
        """
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        N = len(dataset)

        rng = np.random.RandomState(seed)

        scaffolds = defaultdict(list)
        for ind in range(N):
            scaffold = generate_scaffold(dataset[ind]['smiles'], include_chirality=False)
            scaffolds[scaffold].append(ind)

        scaffold_sets = rng.permutation(np.array(list(scaffolds.values()), dtype=object))

        n_total_valid = int(np.floor(frac_valid * len(dataset)))
        n_total_test = int(np.floor(frac_test * len(dataset)))

        train_idx = []
        valid_idx = []
        test_idx = []

        for scaffold_set in scaffold_sets:
            if len(valid_idx) + len(scaffold_set) <= n_total_valid:
                valid_idx.extend(scaffold_set)
            elif len(test_idx) + len(scaffold_set) <= n_total_test:
                test_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        train_dataset = dataset[train_idx]
        valid_dataset = dataset[valid_idx]
        test_dataset = dataset[test_idx]
        return train_dataset, valid_dataset, test_dataset
