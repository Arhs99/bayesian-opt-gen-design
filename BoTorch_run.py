from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from sklearn.metrics import r2_score, mean_absolute_error

import numpy as np
from numpy.random import seed
import pandas as pd

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
from objective import pK

import torch
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement, qUpperConfidenceBound
from botorch.acquisition.analytic import UpperConfidenceBound, ExpectedImprovement, NoisyExpectedImprovement
from botorch.gen import gen_candidates_scipy, get_best_candidates, gen_candidates_torch
from botorch.acquisition.sampler import SobolQMCNormalSampler, IIDNormalSampler
from botorch.optim import joint_optimize, gen_batch_initial_conditions
from botorch.utils.transforms import standardize

MAX_STEPS = 500
# MAX_ITERS = 10
MAX_ITERS_VAL = 10
NUM_TOP = 3
N_DIM = 512
MC_SAMPLES = 500
NUM_RESTARTS = 10
NUM_SAMPLES = 200
NOISE_SE = 0#0.1


def scaleX(X):
    return (X + 1) / 2


def unscale(X):
    return 2 * (X - 1)

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

class BoTorchSingleGP(BOLatentSpace):
    """
        BO experiment class
        """

    def __init__(self, oracle_model_file, oracle_data_file, latent_model_file,
                 init_size, id_field, val_field, smi_field, sampler='temp', thres=0.0, index=list(),
                 acq_type='LCB', kappa=2.0, max_steps=MAX_STEPS, batch_size=1,
                 zero=0.0, use_gpu=True, rand_seed=None):
        # todo: Tidy things up by kwargs

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
                  oracle_model=oracle_model, decoder=decoder, maximize=True)
        self.n_dim = N_DIM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float
        self.domain = torch.tensor([[-1.0] * self.n_dim, [1.0] * self.n_dim], device=self.device, dtype=self.dtype)
        self.batch_size = batch_size

        self.func = func
        self.f = func.f
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

        self.X = torch.tensor(encoder(temp_data), device=self.device, dtype=self.dtype)
        self.Y = vals.reshape(-1) - self.zero
        self.Y = torch.tensor(self.Y, device=self.device, dtype=self.dtype)
        self.best_observed = []
        self.bo_model = SingleTaskGP(train_X=self.X, train_Y=standardize(self.Y))
        # Yvar = torch.tensor(NOISE_SE ** 2, device=self.device, dtype=self.dtype)
        # Yvar = torch.full_like(self.Y, NOISE_SE ** 2)
        # self.bo_model = FixedNoiseGP(self.X, self.Y, Yvar).to(self.X).to(self.X)
        self.mll = ExactMarginalLogLikelihood(self.bo_model.likelihood, self.bo_model).to(self.X)
        self.best_observed.append(standardize(self.Y).max().item())

        self.all_smiles = set(temp_data)
        self.step = 0
        self.max_steps = max_steps
        self.rand_seed = rand_seed
        self.report['SMILES'] = temp_data
        self.report['score'] = self.Y.cpu().numpy()
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

    def _optimize_acqf_and_get_observation(self, acq_func):
        """Optimizes the acquisition function, and returns a new candidate and a noisy observation"""
        batch_initial_conditions = gen_batch_initial_conditions(
            acq_function=acq_func,
            bounds=self.domain,
            q=3,#None, #if isinstance(acq_function, AnalyticAcquisitionFunction) else q,
            num_restarts=NUM_RESTARTS,
            raw_samples=NUM_SAMPLES,
            options={},
        )

        # # optimize using random restart optimization
        # batch_candidates, batch_acq_values = gen_candidates_scipy(
        #     initial_conditions=batch_initial_conditions,
        #     acquisition_function=acq_func,
        #     lower_bounds=self.domain[0],
        #     upper_bounds=self.domain[1],
        #     options={},
        #     # inequality_constraints=inequality_constraints,
        #     # equality_constraints=equality_constraints,
        #     # fixed_features=fixed_features,
        # )

        # torch optimisation
        batch_candidates, batch_acq_values = gen_candidates_torch(
            initial_conditions=batch_initial_conditions,
            acquisition_function=acq_func,
            lower_bounds=self.domain[0],
            upper_bounds=self.domain[1],
            optimizer=torch.optim.Adam,
            verbose=False,
            options={"maxiter": 100},
            # inequality_constraints=inequality_constraints,
            # equality_constraints=equality_constraints,
            # fixed_features=fixed_features,
        )


        candidates = batch_candidates.view(-1, self.n_dim)
        # candidates = get_best_candidates(batch_candidates, batch_acq_values)
        # idx = torch.argsort(batch_acq_values, descending=True)
        # candidates = candidates[idx]

        # candidates = joint_optimize(
        #     acq_function=acq_func,
        #     bounds=self.domain,
        #     q=1,#self.batch_size,
        #     num_restarts=5,
        #     raw_samples=200,
        #     options={}
        # )

        # observe new values
        new_x = candidates.detach()

        # new_x = self._valid(new_x) todo
        new_obj = []
        new_smi = []
        mask = []
        M = -1
        mid = -1
        for i in range(new_x.shape[0]):
            obj, smi = self.f(new_x.cpu().numpy()[i, :])
            if smi is None or len(smi) == 0:
                continue
            else:
                mask.append(i)
                if obj > M:
                    M = obj
                    mid = len(mask) - 1
                new_obj.append(obj)
                new_smi.append(smi)

        if len(mask) == 0:
            return None, 0.0, ''
        else:
            new_x = new_x[mask, :]
            new_obj += NOISE_SE * np.random.randn(len(new_obj))
            return new_x[mid, :].unsqueeze(0), [new_obj[mid]], [new_smi[mid]]

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
            if iters > MAX_ITERS_VAL:
                return x0
            if np.all(x <= 1.0) and np.all(x >= -1):
                smi = self.decoder(x)
                mol = utils.mol_from_xsmiles(smi)
        return x


    def run_bo(self, test_data=None, smiles=None, val=None, max_iter=10):
        """
        Runs the BO
        :param test_data: can provide a test set as a dataframe with a SMILES and real value fields
        :param smiles: SMILES field name in test set if provided
        :param val: value field name in test set if provided
        :param max_iter: Maximum number of iterations that result in unique compound suggestions
        :return:
        """

        self.max_iter = max_iter
        dups = 0
        if self.rand_seed is not None:
            seed(self.rand_seed)
            tf.set_random_seed(self.rand_seed)
            torch.manual_seed(self.rand_seed)
        best_value = self.best_observed[-1]
        while self.cur_iter < self.max_iter:
            fit_gpytorch_model(self.mll)
            # define the qNEI acquisition module using a QMC sampler ## TOO SLOW!
            # qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, seed=self.rand_seed)
            # qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES, resample=True, seed=self.rand_seed)
            qmc_sampler = IIDNormalSampler(MC_SAMPLES, seed=self.rand_seed, resample=True)
            qEI = qExpectedImprovement(model=self.bo_model, sampler=qmc_sampler, best_f=0.0)
            # qnEI = qNoisyExpectedImprovement(model=self.bo_model, sampler=qmc_sampler, X_baseline=self.X)
            # EI = ExpectedImprovement(self.bo_model, best_f=best_value)
            # nEI = NoisyExpectedImprovement(self.bo_model, self.X)
            # UCB = UpperConfidenceBound(self.bo_model, beta=2.0)
            # qUCB = qUpperConfidenceBound(self.bo_model, beta=2.0, sampler=qmc_sampler)

            # optimize and get new observation
            # test = EI(self.X[-7, :].unsqueeze(0))
            # preds = self.bo_model(self.X.unsqueeze(0))
            # preds = self.bo_model.likelihood(self.bo_model(self.X.unsqueeze(0)))
            # print(preds.shape)
            # print(preds.mean.squeeze(0)[-1])
            new_x, new_obj, new_smi = self._optimize_acqf_and_get_observation(qEI)


            self.step += 1
            if self.step > self.max_steps:
                print('Max limit of {} steps exceeded'.format(self.max_steps))
                break

            if new_x is None:
                print(self.step)
                continue

            dist = torch.norm(self.X[-1, :] - new_x, 2).item()
            print(self.step, self.cur_iter, new_smi[0], new_obj[0], dist)
            upd = {'Iteration No': self.cur_iter, 'SMILES': new_smi[0], 'score': new_obj[0] + self.zero}
            # if test_data is not None:
            #     r2, mae = self.metrics(test_data, smiles, val)
            #     upd.update({'R^2': r2, 'MAE': mae})
            self.X = torch.cat((self.X, new_x))
            self.report = self.report.append(upd, ignore_index=True)

            self.Y = torch.cat((self.Y, torch.tensor(new_obj, device=self.device, dtype=self.dtype)))

            uniq_smi = list(filter(lambda x: x not in self.all_smiles, new_smi))

            best_value = standardize(self.Y).max().item()
            # print('Best: ', best_value)
            self.best_observed = best_value
            self.bo_model.set_train_data(self.X, standardize(self.Y), strict=False)

            if len(uniq_smi) != 0:
                self.cur_iter += 1
                self.all_smiles.update(set(uniq_smi))
            else:
                dups += 1
            if dups >= 5 and self.kappa is not None:
                # self.kappa *= 1.5
                dups = 0