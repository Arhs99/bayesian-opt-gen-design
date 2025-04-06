import numpy as np


from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdmolops import GetSymmSSSR
import utils


class ObjectiveFunction(object):
    def __init__(self, data=None, oracle_model=None, decoder=None):
        self.data = data
        self.oracle_model = oracle_model
        self.decoder = decoder

    def f(self, X):
        pass

class pK(ObjectiveFunction):
    def __init__(self, zero, data=None, smiles=None, val=None, oracle_model=None, decoder=None, maximize=False):
        super().__init__(data, oracle_model, decoder)
        if data is not None and (smiles is None or val is None):
            raise ValueError('You need to assign \'smiles\' and \'val\' parameters for data')
        self.zero = zero
        self.val = val
        self.smiles = smiles
        self.hole = 0
        self.sign = 2*maximize - 1

    def f(self, X):  # Returns 2 values so it cannot work if called within GPyOpt
        smi = self.decoder(X)
        mol = utils.mol_from_xsmiles(smi)
        if mol is None:
            return self.sign * (self.hole - self.zero), ''
        can_smi = Chem.MolToSmiles(mol)
        stored_val = self.data[self.data[self.smiles] == can_smi][self.val]
        if not stored_val.empty:
            return self.sign*(stored_val.item() - self.zero), can_smi
        if self.oracle_model is not None:
            fp = utils.morganfp(mol).reshape(1, -1)
            pred = self.oracle_model.predict(fp) - self.zero
            f_x = pred.item() + ring_penalty(mol)
            return self.sign * f_x, can_smi
        return None, can_smi

    def fvec(self, X):
        '''
        Evaluates a NxD matrix X
        :param X:
        :return:
        '''
        f_evals = np.empty(shape=[0, 1])
        for i in range(X.shape[0]):
            rlt = self.f(np.atleast_2d(X[i]))[0]
            if rlt is None:
                print(i, self.decoder(X)[i])
                raise ValueError('Cannot evaluate the objective function possibly an oracle model is missing')
            f_evals = np.vstack([f_evals, rlt])
        return f_evals

def ring_penalty(mol):
    # Penalty should go to acqcuisition function
    penalty = 0
    size_list = [len(x) for x in GetSymmSSSR(mol)]
    for r in size_list:
        if r < 8: continue
        penalty -= r / 8  #
    return min(penalty, 0)
