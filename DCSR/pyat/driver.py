import logging
from pyat.utils.data import AdapTestDataset
from tensorboardX import SummaryWriter
import numpy as np
import random

# 自适应测试的驱动程序,主要功能包括:
#
# 初始化SummaryWriter用于TensorBoard日志记录。
# 在每个测试迭代中:
# 使用策略(strategy)选择题目,更新数据集。
# 使用IRT模型更新参数。
# 计算AUC, coverage等评估指标。
# 将评估结果写入TensorBoard日志。
# 主要依赖AdapTestDataset记录测试数据。
# 每次测试迭代包含:题目选择、模型更新、指标评估、日志记录。
# 可以方便地测试不同策略的效果,通过TensorBoard对比不同策略的评估曲线。
# 主要接口:
# strategy.sel_fisher:策略选择题目
# model.adaptest_update:更新模型
# model.adaptest_evaluate:评估模型
# SummaryWriter:日志记录

class AdapTestDriver(object):

    @staticmethod
    def run(model, strategy, adaptest_data,
            test_length, log_dir):
        writer = SummaryWriter(log_dir)

        logging.info(f'start adaptive testing with {strategy.name} strategy')

        logging.info(f'Iteration 0')
        # evaluate models
        results = model.adaptest_evaluate(adaptest_data)
        # log results
        for name, value in results.items():
            logging.info(f'{name}:{value}')
            writer.add_scalars(name, {strategy.name: value}, 0)

        S_sel = {}
        for sid in range(adaptest_data.num_students):
            key = sid
            S_sel[key] = []
        selected_questions = {}

        for it in range(1, test_length + 1):

            logging.info(f'Iteration {it}')
            if it == 1 and strategy.name == 'BECAT Strategy':
                for sid in range(adaptest_data.num_students):
                    untested_questions = np.array(list(adaptest_data.untested[sid]))
                    random_index = random.randint(0, len(untested_questions) - 1)
                    selected_questions[sid] = untested_questions[random_index]
                    S_sel[sid].append(untested_questions[random_index])
            elif strategy.name == 'BECAT Strategy':
                selected_questions = strategy.adaptest_select(model, adaptest_data, S_sel)
                for sid in range(adaptest_data.num_students):
                    S_sel[sid].append(selected_questions[sid])
            else:
                selected_questions = strategy.adaptest_select(model, adaptest_data)
            for student, question in selected_questions.items():
                adaptest_data.apply_selection(student, question)

                # update models
            model.adaptest_update(adaptest_data)
            # evaluate models
            results = model.adaptest_evaluate(adaptest_data)
            # log results
            for name, value in results.items():
                logging.info(f'{name}:{value}')
                writer.add_scalars(name, {strategy.name: value}, it)

            # # select question
            # logging.info(f'Iteration {it}')
            # selected_questions = strategy.adaptest_select(model, adaptest_data)
            # for student, question in selected_questions.items():
            #     adaptest_data.apply_selection(student, question)
            # # update models
            # model.adaptest_update(adaptest_data)
            # # evaluate models
            # results = model.adaptest_evaluate(adaptest_data)
            # # log results
            # for name, value in results.items():
            #     logging.info(f'{name}:{value}')
            #     writer.add_scalars(name, {strategy.name: value}, it)
