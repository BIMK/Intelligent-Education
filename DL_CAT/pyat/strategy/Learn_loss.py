import numpy as np
from ..strategy.abstract_strategy import AbstractStrategy
import torch

class LALStrategy(AbstractStrategy):

    def __init__(self, n_candidates=10):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def adaptest_select(self, model, net_loss,adaptest_data):
        print(model)
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
                div_loss_list.append(div_loss)

            select_question = untested_questions[div_loss_list.index(max(div_loss_list))]

            selection[sid] = select_question

        return selection






