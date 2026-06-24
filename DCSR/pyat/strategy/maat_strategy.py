import numpy as np
import torch

from pyat.strategy.abstract_strategy import AbstractStrategy
from pyat.model import AbstractModel
from pyat.utils.data import AdapTestDataset

from math import exp as exp
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score
from collections import namedtuple
from scipy.optimize import minimize

from pyat.strategy.NCAT_nn.NCAT import NCATModel

class KLIStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Kullback-Leibler Information Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'get_kli'), \
            'the models must implement get_kli method'
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        n = len(adaptest_data.tested[0])
        for sid in range(adaptest_data.num_students):
            theta = model.get_theta(sid)
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            untested_kli = [model.get_kli(sid, qid, n, pred_all) for qid in untested_questions]
            j = np.argmax(untested_kli)
            selection[sid] = untested_questions[j]
        return selection


class BECATstrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'BECAT Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset, S_set):
        assert hasattr(model, 'delta_q_S_t'), \
            'the models must implement delta_q_S_t method'
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        for sid in range(adaptest_data.num_students):
            tmplen = (len(S_set[sid]))
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            sampled_elements = np.random.choice(untested_questions, tmplen + 5)
            untested_deltaq = [model.delta_q_S_t(qid, pred_all[sid], S_set[sid], sampled_elements) for qid in
                               untested_questions]
            j = np.argmax(untested_deltaq)
            selection[sid] = untested_questions[j]
        # Question bank Q
        return selection


class MAATStrategy(AbstractStrategy):

    def __init__(self, n_candidates=1):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        concept_cnt = {}
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items())
                / sum(1 for c in concept_cnt))

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'expected_model_change'), \
            'the models must implement expected_model_change method'
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            emc_arr = [getattr(model, 'expected_model_change')(sid, qid, adaptest_data)
                       for qid in untested_questions]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            # selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))
            selection[sid] = candidates[0]
        return selection

   
    def sel_fisher(self, model, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = torch.tensor(list(adaptest_data.untested[sid]))

            theta = np.array(model.model.theta(torch.tensor(sid)).detach().numpy()) 
            alpha = np.array(model.model.alpha(untested_questions).detach().numpy()) 
            beta = np.array(model.model.beta(untested_questions).detach().numpy())  
            fisher = self.fisher_information(model, alpha, beta, theta)  
            selection[sid] = untested_questions[np.argmax(fisher)].item()
        return selection

    
    def fisher_information(self, model, alpha, beta, theta):
        """ calculate the fisher information
        """
        try:
            information = []
            for t in theta:
                p = model.irf(alpha, beta, t)  
                q = 1 - p  
                pdt = model.pd_irf_theta(alpha, beta, t) 
                # information.append((pdt**2) / (p * q))
                information.append(p * q * (alpha ** 2))
            information = np.array(information)
            return information
        except TypeError:
            p = model.irf(alpha, beta, theta)
            q = 1 - p
            # pdt = model.pd_irf_theta(alpha, beta, theta)
            # return (pdt ** 2) / (p * q + 1e-7)
            return (p * q * (alpha ** 2))


class NCATs(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'NCAT'

    def adaptest_select(self, adaptest_data: AdapTestDataset,concept_map,config,test_length):
        selection = {}
        NCATdata = adaptest_data
        model = NCATModel(NCATdata,concept_map,config,test_length)
        threshold = config['THRESHOLD']
        for sid in range(adaptest_data.num_students):
            print(str(sid+1)+'/'+str(adaptest_data.num_students))
            used_actions = []
            model.ncat_policy(sid,threshold,used_actions,type="training",epoch=100)
        NCATdata.reset()
        for sid in range(adaptest_data.num_students):
            used_actions = []
            model.ncat_policy(sid,threshold,used_actions,type="testing",epoch=0)
            selection[sid] = used_actions
        NCATdata.reset()
        return selection
