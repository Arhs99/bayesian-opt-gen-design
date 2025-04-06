import utils
from objective import pK

import gpflow
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.acquisition import ExpectedImprovement, LowerConfidenceBound
from gpflowopt.optim import SciPyOptimizer, StagedOptimizer, MCOptimizer
from gpflowopt.domain import ContinuousParameter, Domain
from gpflow import settings

import numpy as np
from numpy.random import seed
from sklearn.metrics import r2_score, mean_absolute_error

import pandas as pd
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit.Chem import AllChem as Chem

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MAX_STEPS = 500
MAX_ITERS = 10
NUM_TOP = 3
N_DIM = 512
LIK_VAR = 0.01
OPT_RESTARTS = 5
NOISE_SE = 0.1

class BOLatentSpace:
    '''
    Interface for BO experiments
    '''
    #todo: Add a method to save model
    def __init__(self, bo_model, oracle_model, oracle_data, encoder, decoder, zero):
        self.bo_model = bo_model
        self.encoder = encoder
        self.decoder = decoder
        self.report = pd.DataFrame(columns=['Iteration No', 'SMILES', 'score', 'R^2', 'MAE'])
        self.oracle_data = oracle_data
        self.oracle_model = oracle_model
        self.zero = zero

    def run_bo(self):
        pass

    def suggest(self, batch_num=10):
        pass

    def output_results(self):
        return self.report

    def metrics(self, test_data, smiles, val):
        """
        If a test set is provided, return R^2 and MAE
        :param test_data:
        :param smiles:
        :param val:
        :return:
        """
        if self.bo_model is None:
            raise ValueError('A BO model needs to be initialized first')
        X_test, y_test = utils.extract_XY(test_data, self.encoder, smiles=smiles, val=val)
        y_pred, _ = self.bo_model.model.predict(X_test)
        y_pred = - y_pred + self.zero
        r2 = r2_score(y_true=y_test, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        return r2, mae

class GPFOptBatchOne(BOLatentSpace):
    """
    BO experiment class
    """
    def __init__(self, oracle_model_file, oracle_data_file, latent_model_file,
                 init_size, id_field, val_field, smi_field, sampler='temp', thres = 0.0, index=list(),
                 acq_type='LCB', kappa=20.0, max_steps=MAX_STEPS,
                 zero=0.0, use_gpu=True, rand_seed=None):
        #todo: Tidy things up by kwargs

        oracle_model = utils.oracle_model_loader(oracle_model_file)
        oracle_data = utils.oracle_data_loader(oracle_data_file)
        oracle_data[smi_field] = oracle_data[smi_field].apply(
            lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False)
            if Chem.MolFromSmiles(x) is not None else None)
        oracle_data = oracle_data.dropna()
        latent_model = utils.latent_model_load(latent_model_file, use_gpu=use_gpu, num_top=NUM_TOP)
        encoder = latent_model.seq_to_emb
        decoder = latent_model.emb_to_seq
        func = pK(zero=zero, data=oracle_data, smiles='SMILES', val=val_field,
                  oracle_model=oracle_model, decoder=decoder) if oracle_model_file else None
        self.n_dim = N_DIM
        # params = [ContinuousParameter('x{0}'.format(i + 1), -1.0, 1.0) for i in np.arange(self.n_dim)]
        # self.domain = Domain(params)
        lower = [-1.] * self.n_dim
        upper = [1.] * self.n_dim
        self.domain = np.sum([ContinuousParameter('x{0}'.format(i+1), l, u)
                              for i, l, u in zip(range(self.n_dim), lower, upper)])

        self.func = func
        self.cur_iter = 1
        self.acq_type = acq_type
        self.kappa = kappa
        self.id_field = id_field
        self.smi_field = smi_field
        self.val_field = val_field
        self.thres = thres
        self.index = index
        bo_model = None
        self.init_size = init_size

        super().__init__(bo_model, oracle_model, oracle_data, encoder, decoder, zero)

        temp_data, vals = self._choose_sampler(sampler)
        self.init_size = len(temp_data)
        if self.init_size != init_size:
            print('init_size={} is ignored. Using init_size={} instead'.format(init_size, self.init_size))
        self.X = encoder(temp_data)
        # self.Y = self._eval_func(self.X)
        self.Y = -vals + self.zero
        # self.Y += NOISE_SE * np.random.randn(self.Y.shape[0])
        self.all_smiles = set(temp_data)
        self.step = 0
        self.max_steps = max_steps
        self.rand_seed = rand_seed
        self.report.SMILES = temp_data
        self.report.score = -1.0 * self.Y
        self.report['Iteration No'] = 0


    def _choose_sampler(self, sampler):
        if sampler == 'temp':
            return utils.init_bo_temporal(self.oracle_data, self.init_size,
                                                 id_field=self.id_field, smi_field=self.smi_field, val=self.val_field)
        elif sampler == 'random':
            return utils.init_bo_set(self.oracle_data, length=self.init_size,
                                            thres=self.thres, smi_field=self.smi_field, val=self.val_field)
        elif sampler == 'indexed':
            if len(self.index) > 0:
                return utils.init_indexed_set(self.oracle_data, index=self.index,
                                          smi_field=self.smi_field, val=self.val_field)
            else:
                raise ValueError('Index list cannot be empty')
        else:
            raise ValueError('\"{}\" option for sampler is not available'.format(sampler))


    def suggest(self, batch_num=10, sampling='neighbors', unique=False, **kwargs):
        """
        Provides an initial suggestion from BO and then samples the space around this to provide additional suggestions
        Running this multiple times in a stochastic manner (e.g. with different random seed values) can provide
        suggestions
        :param batch_num: Batch size, number of suggestions can be lower
        :param sampling: only 'neighbors' currently available
        :return: dataframe
        """
        if sampling == 'neighbors':
            return self._custom_suggest(batch_num, unique, **kwargs)
        # elif sampling == 'single':
        #     return self._single_suggest(batch_num, unique, kwargs)
        else:
            raise ValueError('Sampling method \'{}\' is not currently implemented'.format(sampling))


    def _custom_suggest(self, num_n=1, unique=False, **kwargs):
        def f(X):
            ret = np.ones((X.shape[0], 1)) * -9.0
            return ret
        if self.rand_seed is not None:
            seed(self.rand_seed)
            tf.set_random_seed(self.rand_seed)
        # Introduce stochasticity by adding noise to the y's
        Y = self.Y.copy() + NOISE_SE * np.random.randn(*self.Y.shape)
        self.bo_model = gpflow.gpr.GPR(self.X.copy(), Y,
                                       gpflow.kernels.Matern52(self.n_dim, ARD=False))
        self.bo_model.likelihood.variance = LIK_VAR
        alpha = self._choose_acq_function()
        acquisition_opt = StagedOptimizer([MCOptimizer(self.domain, 2000),
                                           SciPyOptimizer(self.domain)])
        optimizer = BayesianOptimizer(self.domain, alpha, optimizer=acquisition_opt, scaling=True, verbose=True)

        func = self.func.fvec if self.func else f
        with optimizer.silent():
            r = optimizer.optimize(func, n_iter=1)

        x0 = optimizer.acquisition.data[0][-1, :]

        if num_n == 1:
            smi = self.decoder(x0)
            mol = utils.mol_from_xsmiles(smi)
            smi = Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else ''
            if len(smi) > 0:
                posterior_means, posterior_vars = alpha.models[0].predict_f(x0.reshape(1, -1))
                return smi, -posterior_means.item(), np.sqrt(posterior_vars.item())
            else:
                return '', -1, -1

        x0 = self._valid(x0)
        X, smis = utils.unique_var_neighbors(x0, self.decoder, self.n_dim, max_dist=kwargs['max_dist'],
                                             num=num_n, timer=kwargs['timer'])

        posterior_means, posterior_vars = alpha.models[0].predict_f(X)
        posterior_stds = np.sqrt(posterior_vars)
        posterior_means = self.zero - posterior_means
        df = pd.DataFrame({'SMILES': smis, 'posterior E[X]': posterior_means.flatten(),
                           'posterior std': posterior_stds.flatten()})

        self.bo_model.randomize()
        if unique:
            df = df[~df['SMILES'].isin(self.all_smiles)]
        return df

    def _choose_acq_function(self):
        if self.acq_type == 'EI':
            return ExpectedImprovement(self.bo_model)
        elif self.acq_type == 'LCB':
            return LowerConfidenceBound(self.bo_model, sigma=self.kappa)
        else:
            raise ValueError('Acquisition function \'{}\' is not currently implemented'.format(self.acq_type))


    def run_bo(self, test_data=None, smiles=None, val=None, max_iter=10):
        self.max_iter = max_iter
        dups = 0
        if self.rand_seed is not None:
            seed(self.rand_seed)
            tf.set_random_seed(self.rand_seed)
        while self.cur_iter < self.max_iter:
            if self.step > self.max_steps:
                print('Max limit of {} steps exceeded'.format(self.max_steps))
                break
            self.bo_model = gpflow.gpr.GPR(self.X.copy(), self.Y.copy(), gpflow.kernels.Matern52(self.n_dim, ARD=False))
            self.bo_model.likelihood.variance = LIK_VAR
            # alpha = LowerConfidenceBound(self.bo_model, sigma=self.kappa) # Extend this with an acq chooser function
            alpha = self._choose_acq_function()
            acquisition_opt = StagedOptimizer([MCOptimizer(self.domain, 2000),
                                               SciPyOptimizer(self.domain)])
            optimizer = BayesianOptimizer(self.domain, alpha, optimizer=acquisition_opt, scaling=True, verbose=True)
            optimizer.acquisition.optimize_restarts = OPT_RESTARTS
            with optimizer.silent():
                r = optimizer.optimize(self.func.fvec, n_iter=1)
            x_next = optimizer.acquisition.data[0][-1, :]
            x_next = self._valid(x_next)
            y_next, smi_next = self.func.f(x_next)
            if y_next is None:
                raise ValueError('Cannot evaluate the objective function possibly an oracle model is missing')
            self.step += 1

            dist = np.sqrt(np.sum((self.X[-1, :] - x_next) ** 2))
            if len(smi_next) > 0:
                posterior_means, posterior_vars = alpha.models[0].predict_f(x_next.reshape(1, -1))
                # return smi, self.zero - posterior_means.item(), np.sqrt(posterior_vars.item())
                print('Step:{}\tIteration:{}\t{}\t\t\ty={:.1f}\tdist={:.2f}\tE[X]={:.3f}'.
                      format(self.step, self.cur_iter, smi_next, -y_next, dist, posterior_means.item()))

            upd = {'Iteration No': self.cur_iter, 'SMILES': smi_next, 'score': -y_next + self.zero}
            if test_data is not None:
                r2, mae = self.metrics(test_data, smiles, val)
                upd.update({'R^2': r2, 'MAE': mae})
            self.X = np.vstack((self.X, x_next))
            self.report = self.report.append(upd, ignore_index=True)
            self.Y = np.vstack((self.Y, y_next))

            if smi_next not in self.all_smiles:
                self.cur_iter += 1
                self.all_smiles.add(smi_next)
            else:
                dups += 1
            if dups >= 5 and self.kappa is not None:
                # self.kappa *= 1.5
                dups = 0


    def _valid(self, x0):
        """
        if x0 decodes to a valid molecule return x0 otherwise sample from the neighborhood of x0 with growing radius
        until a valid x0 is found or MAX_ITERS limit is reached
        """
        iters = 0
        r = 1
        smi = self.decoder(x0)
        mol = utils.mol_from_xsmiles(smi)
        x = x0.copy()
        while (mol is None) or (len(smi) == 0):
            v = np.random.uniform(-1, 1, (1, self.n_dim))
            v = r * (v / np.linalg.norm(v))
            x = x0 + v
            x -= 2 * (x > 1.0) * v
            x -= 2 * (x < -1.0) * v
            iters += 1
            if iters % 10 == 0:
                r += 0.5
            if iters > MAX_ITERS:
                return x0
            if np.all(x <= 1.0) and np.all(x >= -1):
                smi = self.decoder(x)
                mol = utils.mol_from_xsmiles(smi)
        return x


    def _eval_func(self, x):
        """
        Performs sequential evaluations of the function at x (single location or batch). The computing time of each
        evaluation is also provided.
        """
        f_evals = np.empty(shape=[0, 1])
        for i in range(x.shape[0]):
            rlt = self.f(np.atleast_2d(x[i]))[0]
            if rlt is None:
                print(i, self.decoder(x)[i])
                raise ValueError('Cannot evaluate the objective function possibly an oracle model is missing')
            f_evals  = np.vstack([f_evals, rlt])
        return f_evals
