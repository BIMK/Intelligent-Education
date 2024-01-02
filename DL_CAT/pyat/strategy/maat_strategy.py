import numpy as np
from ..strategy.abstract_strategy import AbstractStrategy
from ..model import AbstractModel
from ..utils.data import AdapTestDataset
import random
import torch
import heapq

class MAATStrategy(AbstractStrategy):

    def __init__(self, n_candidates=10):
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

    def adaptest_select(self, model: AbstractModel, net_loss, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'expected_model_change'), \
            'the models must implement expected_model_change method'
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            emc_arr = [getattr(model, 'expected_model_change')(sid, qid, adaptest_data)
                       for qid in untested_questions]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            # selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))
            selection[sid] = max(candidates)
        return selection

    def select_emc_cov(self, model: AbstractModel, net_loss, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            emc_arr = model.select_emc(sid, untested_questions, adaptest_data._knowledge_embs)
            # questions = untested_questions[emc_arr.index(max(emc_arr))]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))

        return selection

    def select_emc(self, model: AbstractModel, net_loss, adaptest_data: AdapTestDataset):
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            emc_arr = model.select_emc(sid, untested_questions, adaptest_data._knowledge_embs)
            questions = untested_questions[emc_arr.index(max(emc_arr))]
            selection[sid] = questions
        return selection


    def sel_rand(self, model, adaptest_data:AdapTestDataset):
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = list(adaptest_data.untested[sid])
            selection[sid] = random.choice(untested_questions)
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
                information.append(p * q * (alpha**2))
            information = np.array(information)
            return information
        except TypeError:
            p = model.irf(alpha, beta, theta)
            q = 1 - p
            # pdt = model.pd_irf_theta(alpha, beta, theta)
            # return (pdt ** 2) / (p * q + 1e-7)
            return (p * q * (alpha**2))

    def DLCAT_select_irt(self, model, net_loss, adaptest_data):
        selection = {}
        for sid in range(adaptest_data.num_students):
            # 获得theta参数
            div_loss_list = []
            theta = model.model.theta.weight[sid]
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            for qid in untested_questions:
                alpha = model.model.alpha.weight[qid]
                beta = model.model.beta.weight[qid]
                param = torch.cat((theta, alpha, beta), 0)
                div_loss = net_loss.forward(param)
                div_loss = div_loss.item()
                div_loss_list.append(div_loss)

            select_question = untested_questions[div_loss_list.index(max(div_loss_list))]

            selection[sid] = select_question

        return selection

    def DLCAT_select_irt1(self, model, net_loss, adaptest_data):
        selection = {}
        for sid in range(adaptest_data.num_students):
            # 获得theta参数
            div_loss_list = []
            theta = model.model.theta.weight[sid]
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            for j in range(len(untested_questions)):
                qid = untested_questions[j]
                if j == 0:
                    alpha = model.model.alpha.weight[qid]
                    beta = model.model.beta.weight[qid]
                    param = torch.cat((theta, alpha, beta), 0)
                    param = param.unsqueeze(0)
                else:
                    alpha = model.model.alpha.weight[qid]
                    beta = model.model.beta.weight[qid]
                    add_param = torch.cat((theta, alpha, beta), 0)
                    # 升维
                    add_param = add_param.unsqueeze(0)
                    param = torch.cat((param, add_param), dim=0)

            div_loss = net_loss.forward(param).tolist()
            select_question = untested_questions[div_loss.index(max(div_loss))]

            selection[sid] = select_question

        return selection


