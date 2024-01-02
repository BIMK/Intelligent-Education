import logging
from .utils.data import AdapTestDataset
from tensorboardX import SummaryWriter
import numpy as np

ran_auc_list = []
ran_acc_list = []
fisher_auc_list = []
fisher_acc_list = []
Maat_auc_list = []
Maat_acc_list = []
Maat_cov_auc_list = []
Maat_cov_acc_list = []
DLCAT_auc_list = []
DLCAT_acc_list = []
DLCAT_cov_auc_list = []
DLCAT_cov_acc_list = []
real_auc_list = []
real_acc_list = []

random_rmse_list = []
fisher_rmse_list = []
Maat_rmse_list = []
DLCAT_rmse_list = []
Maat_cov_rmse_list = []

class AdapTestDriver(object):

    @staticmethod
    def run(model, net_loss, strategy, adaptest_data, test_length, log_dir, sel, rcd_model):
        writer = SummaryWriter(log_dir)

        logging.info(f'start adaptive testing with {sel} strategy')
        logging.info(f'Iteration 0')
        # evaluate models
        results = model.adaptest_evaluate(adaptest_data)
        # log results
        for name, value in results.items():
            logging.info(f'{name}:{value}')
            writer.add_scalars(name, {strategy.name: value}, 0)

        # print('start_theta', model.model.theta.weight.data[:5])

        for it in range(1, test_length + 1):
            logging.info(f'Iteration {it}')
            if sel =='random':
                selected_questions = strategy.sel_rand(model, adaptest_data)
            elif sel =='fisher':
                selected_questions = strategy.sel_fisher(model, adaptest_data)
            elif sel == 'Maat':
                selected_questions = strategy.select_emc(model, net_loss, adaptest_data)
            elif sel == 'Maat_cov':
                selected_questions = strategy.select_emc_cov(model, net_loss, adaptest_data)
            elif sel =='Maat_real':
                selected_questions = strategy.adaptest_select(model, net_loss, adaptest_data)
            elif sel =='DLCAT':
                selected_questions = strategy.DLCAT_select_irt(model, net_loss, adaptest_data)


            for student, question in selected_questions.items():
                adaptest_data.apply_selection(student, question)
            # update models
            model.adaptest_update(adaptest_data)
            # evaluate models
            results = model.adaptest_evaluate(adaptest_data)

            # log results
            for name, value in results.items():
                logging.info(f'{name}:{value}')
                # writer.add_scalars(name, {strategy.name: value}, it)

            if sel == 'random':
                ran_auc_list.append(results['auc'])
                ran_acc_list.append(results['acc'])
            elif sel == 'fisher':
                fisher_auc_list.append(results['auc'])
                fisher_acc_list.append(results['acc'])
            elif sel == 'Maat':
                Maat_auc_list.append(results['auc'])
                Maat_acc_list.append(results['acc'])
            elif sel == 'Maat_cov':
                Maat_cov_auc_list.append(results['auc'])
                Maat_cov_acc_list.append(results['acc'])
            elif sel == 'DLCAT':
                DLCAT_auc_list.append(results['auc'])
                DLCAT_acc_list.append(results['acc'])
            elif sel == 'DLCAT_cov':
                DLCAT_cov_auc_list.append(results['auc'])
                DLCAT_cov_acc_list.append(results['acc'])

            elif sel == 'real':
                real_auc_list.append(results['auc'])
                real_acc_list.append(results['acc'])

        # return ran_auc_list, fisher_auc_list, Maat_auc_list, Maat_cov_auc_list, DLCAT_auc_list, DLCAT_cov_auc_list
        return ran_auc_list, fisher_auc_list, Maat_auc_list, Maat_cov_auc_list, DLCAT_auc_list, DLCAT_cov_auc_list, ran_acc_list, fisher_acc_list, Maat_acc_list, Maat_cov_acc_list,  DLCAT_acc_list, real_auc_list, real_acc_list



    def run_real( model, net_loss, strategy, adaptest_data, test_length, log_dir, sel, rcd_model,real_theta):
        writer = SummaryWriter(log_dir)

        logging.info(f'start adaptive testing with {sel} strategy')
        logging.info(f'Iteration 0')
        # evaluate models
        results = model.adaptest_evaluate(adaptest_data)
        # log results
        for name, value in results.items():
            logging.info(f'{name}:{value}')
            writer.add_scalars(name, {strategy.name: value}, 0)

        # print('start_theta', model.model.theta.weight.data[:5])

        for it in range(1, test_length + 1):
            logging.info(f'Iteration {it}')
            if sel =='random':
                selected_questions = strategy.sel_rand(model, adaptest_data)
            elif sel =='fisher':
                selected_questions = strategy.sel_fisher(model, adaptest_data)
            elif sel == 'Maat':
                selected_questions = strategy.select_emc(model, net_loss, adaptest_data)
            elif sel == 'Maat_cov':
                selected_questions = strategy.select_emc_cov(model, net_loss, adaptest_data)
            elif sel =='Maat_real':
                selected_questions = strategy.adaptest_select(model, net_loss, adaptest_data)
            elif sel =='DLCAT':
                if rcd_model == 'irt' or rcd_model == 'mirt':
                    selected_questions = strategy.DLCAT_select_irt(model, net_loss, adaptest_data)
                elif rcd_model == 'ncd':
                    selected_questions = strategy.DLCAT_select_ncd(model, net_loss, adaptest_data)
            elif sel == 'DLCAT_cov':
                selected_questions = strategy.DLCAT_select_ncd_cov(model, net_loss, adaptest_data)

            elif sel == 'real':
                selected_questions = strategy.real_select(model, net_loss, adaptest_data)

            # print('after_theta', model.model.theta.weight.data[:5])

            for student, question in selected_questions.items():
                adaptest_data.apply_selection(student, question)
            # update models
            model.adaptest_update(adaptest_data)


            # 计算real_theta和选题后更新的theta的rmse
            if sel =='random':
                rmse = np.sqrt(np.mean(np.square(np.array(real_theta.tolist()) - np.array(model.model.theta.weight.tolist()))))
                random_rmse_list.append(rmse)
            elif sel == 'fisher':
                rmse = np.sqrt(np.mean(np.square(np.array(real_theta.tolist()) - np.array(model.model.theta.weight.tolist()))))
                fisher_rmse_list.append(rmse)
            elif sel == 'Maat':
                rmse = np.sqrt(np.mean(np.square(np.array(real_theta.tolist()) - np.array(model.model.theta.weight.tolist()))))
                Maat_rmse_list.append(rmse)
            elif sel == 'Maat_cov':
                rmse = np.sqrt(np.mean(np.square(np.array(real_theta.tolist()) - np.array(model.model.theta.weight.tolist()))))
                Maat_cov_rmse_list.append(rmse)
            elif sel == 'DLCAT':
                rmse = np.sqrt(np.mean(np.square(np.array(real_theta.tolist()) - np.array(model.model.theta.weight.tolist()))))
                DLCAT_rmse_list.append(rmse)

            # evaluate models
            results = model.adaptest_evaluate(adaptest_data)

            # log results
            for name, value in results.items():
                logging.info(f'{name}:{value}')
                # writer.add_scalars(name, {strategy.name: value}, it)

            if sel == 'random':
                ran_auc_list.append(results['auc'])
                ran_acc_list.append(results['acc'])
            elif sel == 'fisher':
                fisher_auc_list.append(results['auc'])
                fisher_acc_list.append(results['acc'])
            elif sel == 'Maat':
                Maat_auc_list.append(results['auc'])
                Maat_acc_list.append(results['acc'])
            elif sel == 'Maat_cov':
                Maat_cov_auc_list.append(results['auc'])
                Maat_cov_acc_list.append(results['acc'])
            elif sel == 'DLCAT':
                DLCAT_auc_list.append(results['auc'])
                DLCAT_acc_list.append(results['acc'])
            elif sel == 'DLCAT_cov':
                DLCAT_cov_auc_list.append(results['auc'])
                DLCAT_cov_acc_list.append(results['acc'])

            elif sel == 'real':
                real_auc_list.append(results['auc'])
                real_acc_list.append(results['acc'])

        # return ran_auc_list, fisher_auc_list, Maat_auc_list, Maat_cov_auc_list, DLCAT_auc_list, DLCAT_cov_auc_list
        return ran_auc_list, fisher_auc_list, Maat_auc_list, Maat_cov_auc_list, DLCAT_auc_list, DLCAT_cov_auc_list, ran_acc_list, fisher_acc_list, Maat_acc_list, Maat_cov_acc_list,  DLCAT_acc_list, real_auc_list, real_acc_list,random_rmse_list,fisher_rmse_list, Maat_rmse_list, Maat_cov_rmse_list,DLCAT_rmse_list


