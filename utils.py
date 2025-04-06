import pandas as pd
import numpy as np
from numpy.random import seed

from cddd.inference import InferenceModel

from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import DataStructs, Descriptors
from sklearn.externals import joblib

import os

CHEMBL_DATA_FILE = '/home/UK/kpapadop/DeepLearning/molecule_generator/datasets/ChEMBL.txt'


def morganfp(mol, bits=2048, radius=2):
    vec = np.ndarray((1, bits), dtype=int)
    fp = Chem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
    DataStructs.ConvertToNumpyArray(fp, vec)
    return vec


def init_bo_set(data, length, val, thres, smi_field='SMILES', ratio_actives=0.25, s=None):
    """
    Produces from data a sample with a defined size and ratio of actives/inactives
    :param data:
    :param length:
    :param val:
    :param thres:
    :param smi_field:
    :param ratio_actives:
    :param s:
    :return: a tuple of smiles and property values
    """
    if s: seed(s)
    idx_inact = data[data[val] < thres].index
    idx_act = data[data[val] >= thres].index
    num_actives = int(length * ratio_actives)
    num_inactives = length - num_actives
    act_arr = np.random.choice(idx_act, num_actives, replace=False)
    inact_arr = np.random.choice(idx_inact, num_inactives, replace=False)
    combi = np.hstack((act_arr, inact_arr))
    np.random.shuffle(combi)
    smi = data[smi_field][combi].tolist()
    vals = data[val][combi].values.reshape(-1, 1)
    return smi, vals


def init_indexed_set(data, val, index, smi_field='SMILES'):
    smi = data[smi_field][index].tolist()
    vals = data[val][index].values.reshape(-1, 1)
    return smi, vals


def extract_XY(data, encoder, smiles=None, val=None):
    if data is not None and (smiles is None or val is None):
        raise ValueError('You need to assign \'smiles\' and \'val\' parameters for data')
    smis = data[smiles].tolist()
    smis = [Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False) for x in smis if x is not None]
    X_test = encoder(smis)
    y_test = data[val].values
    return X_test, y_test


def init_bo_from_chembl(length):
    with open(CHEMBL_DATA_FILE, 'r') as f:
        chembl_data = np.random.choice(f.readlines(), length, replace=False)
    return [Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False) for x in chembl_data.tolist() if x is not None]


def init_bo_temporal(data, length, id_field, smi_field='SMILES', val=None):
    """
    Sorts the initial set by ID assuming that: smaller ID -> earlier time,
    then returns required slice
    :param smi_field:
    :param data:
    :param length:
    :return:
    """
    temp_data = data.sort_values(id_field).iloc[:length, :].copy()
    smi = temp_data[smi_field].tolist()
    vals = temp_data[val].values.reshape(-1, 1)
    # canon = [Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False) for x in L if x is not None]
    return smi, vals


def sample_from_evaluations(decoder, bo_model, init_size, threshold=0.9):
    X, y = bo_model.get_evaluations()
    X = X[init_size:, :]
    y = y[init_size:, :]
    idx = np.where( y <= -threshold)[0] # given that we minimize -f
    X_filt = X[idx, :]
    if len(X_filt) == 0: return
    smi_list = decoder(X_filt)
    mols = [Chem.MolFromSmiles(x) for x in smi_list if x is not None]
    return mols


def dedup_mols(mols):
    smi = set(Chem.MolToSmiles(x) for x in mols)
    return [Chem.MolFromSmiles(x) for x in smi if x is not None]


def oracle_model_loader(filename):
    if filename is None:
        return None
    model = joblib.load(filename)
    print('Oracle model loaded')
    return model


def oracle_data_loader(filename):
    data = pd.read_csv(filename)
    print('Oracle data loaded')
    return data


def latent_model_load(model_dir, use_gpu=True, device=0, num_top=1):
    if use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    model = InferenceModel(model_dir, use_gpu=use_gpu, num_top=num_top)
    print('Latent space model loaded')
    return model


def mol_from_xsmiles(smis):
    if isinstance(smis, str):
        return Chem.MolFromSmiles(smis)
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return mol
    return None


def unique_var_neighbors(x0, decoder, dim, max_dist, num=1, timer=100):
    n = 0
    X = x0.copy()
    smis = set()
    smi0 = decoder(x0)
    mol0 = mol_from_xsmiles(smi0)
    if mol0 is not None:
        smi0 = Chem.MolToSmiles(mol0, isomericSmiles=False)
        smis.add(smi0)
    ret_smis = [smi0]
    t = 0
    while len(ret_smis) <= num+1 and t < timer:
        v = np.random.uniform(-1, 1, (1, dim))
        r = max((1 - np.random.rand()) * max_dist, 4)    # b.c np.rand() is in [0, 1), we want (0, 1]
        v = r * (v / np.linalg.norm(v))
        x = x0 + v
        x -= 2 * (x > 1.0) * v
        x -= 2 * (x < -1.0) * v
        t += 1
        if np.all(x <= 1.0) and np.all(x >= -1):
            new_smi = decoder(x)
            new_mol = mol_from_xsmiles(new_smi)
            if new_mol is not None:
                smi = Chem.MolToSmiles(new_mol, isomericSmiles=False)
                if smi not in smis:
                    if len(smi):
                        ret_smis.append(smi)
                        X = np.vstack((X, x))
                smis.add(smi)
    print(t, 'iterations')
    return X, ret_smis