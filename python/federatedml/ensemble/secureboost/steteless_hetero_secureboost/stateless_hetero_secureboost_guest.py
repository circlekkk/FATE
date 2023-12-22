from federatedml.ensemble.secureboost.steteless_hetero_secureboost.stateless_hetero_secureboost_base import StatelessHeteroSecureBoostingTreeBase
class StatelessHeteroSecureBoostingTreeGuest(StatelessHeteroSecureBoostingTreeBase):

    def __init__(self):
        super(StatelessHeteroSecureBoostingTreeGuest, self).__init__()
        self.data_instance_weight=None#
        self.loss_method=self.CrossEntropyLoss()
    def compute_grad_and_hess(self,y,y_hat,data):
        pass



