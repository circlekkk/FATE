from federatedml.ensemble.secureboost.steteless_hetero_secureboost.stateless_hetero_secureboost_base import \
    StatelessHeteroSecureBoostingTreeBase
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.loss import SigmoidBinaryCrossEntropyLoss
from federatedml.ensemble.secureboost.steteless_hetero_secureboost import stateless_hetero_secureboost_consts
import pandas as pd
import numpy as np


# class StatelessHeteroSecureBoostingTreeGuest(StatelessHeteroSecureBoostingTreeBase):
#
#     def __init__(self, job_id=None,sample_data_set_with_label_path=None,loss_method=SigmoidBinaryCrossEntropyLoss,
#                  task_type="classification"):
#         super(StatelessHeteroSecureBoostingTreeGuest, self).__init__()
#         # self.data_instance_weight_path = data_instance_weight_path  # 每个样本的权重的csv，暂时不用这个参数
#         self.loss_method = loss_method
#         self.task_type = task_type
#         self.max_sample_weight_computed=False
#         self.max_sample_weight=1
#         self.sample_data_set_with_label_path=sample_data_set_with_label_path
#         self.job_id=job_id
#         self.inital_y_and_y_hat_csv()
def inital_y_and_y_hat_csv(job_id, sample_data_set_with_label_path):
    sample_data_set_with_label = pd.read_csv(sample_data_set_with_label_path)
    y = sample_data_set_with_label['y']
    y.to_csv(f'{job_id}_guest_y.csv', index=False, header=['y'])
    sample_data_set_num_rows = sample_data_set_with_label.shape[0]
    y_hat_initial_data = np.random.rand(sample_data_set_num_rows).round(6)
    y_hat_initial_data = pd.DataFrame(y_hat_initial_data)
    y_hat_initial_data.to_csv(f'{job_id}_guest_y_hat.csv', index=False, header=['y_hat'])


def compute_grad_and_hess(job_id, y_path, y_hat_path, task_type, loss_method):
    y = pd.read_csv(y_path)['y']
    y_hat = pd.read_csv(y_hat_path)['y_hat']
    if task_type == stateless_hetero_secureboost_consts.CLASSIFICATION:
        sigmod_y_hat = y_hat.apply(lambda y_hat_val: loss_method.predict(y_hat_val))
        grad = pd.Series(map(loss_method.compute_grad, y, sigmod_y_hat), name='grad')
        hess = pd.Series(map(loss_method.compute_hess, y, sigmod_y_hat), name='hess')
        # grad_and_hess=pd.concat([grad,hess],axis=1)
        # grad_and_hess.to_csv(f'{self.job_id}_guest_grad_and_hess.csv',index=False)
    else:
        grad = pd.Series(map(loss_method.compute_grad, y, y_hat), name='grad')
        hess = pd.Series(map(loss_method.compute_hess, y, y_hat), name='hess')
    grad_and_hess = pd.concat([grad, hess], axis=1)
    grad_and_hess.to_csv(f'{job_id}_guest_grad_and_hess.csv', index=False)
    # grad_and_hess = self.process_sample_weights(grad_and_hess)
    # return grad_and_hess


def process_sample_weights(grad_and_hess_path, data_instance_weight_path=None):
    """
    如果样本权重不为空，g和h乘样本权重
    覆盖{job_id}_guest_grad_and_hess.csv
    @param grad_and_hess_path:
    @param data_instance_weight_path:
    """
    if data_instance_weight_path is not None:
        data_instance_weight = pd.read_csv(data_instance_weight_path)['sample_weight']
        grad_and_hess = pd.read_csv(grad_and_hess_path)
        result = grad_and_hess.multiply(data_instance_weight, axis='rows')
        result.to_csv(f'{job_id}_guest_grad_and_hess.csv', index=False)
        #####这里暂时搁置
        # max_sample_weight_computed=False
        # max_sample_weight=1
        # if not max_sample_weight_computed:
        #     # 如果没有计算过最大样本权重
        #     max_sample_weight = get_max_sample_weight(data_with_sample_weight)
        #     max_sample_weight_computed = True

def on_epoch_prepare(epoch_idx):
    """
    Prepare g, h, sample weights, sampling at the beginning of every epoch
    @param epoch_idx: 当前的epoch_id
    """


if __name__ == '__main__':
    sample_data_set_with_label_path = 'stateless_guest_data.csv'
    weight_path = 'stateless_sample_weight.csv'
    y_path = '1001_guest_y.csv'
    y_hat_path = '1001_guest_y_hat.csv'
    job_id = '1001'
    grad_and_hess_path = '1001_guest_grad_and_hess.csv'
    loss_method = SigmoidBinaryCrossEntropyLoss
    task_type = "classification"
    # inital_y_and_y_hat_csv(job_id, sample_data_set_with_label_path)
    # compute_grad_and_hess(job_id,y_path, y_hat_path, task_type, loss_method)
    process_sample_weights(data_instance_weight_path=weight_path, grad_and_hess_path=grad_and_hess_path)
