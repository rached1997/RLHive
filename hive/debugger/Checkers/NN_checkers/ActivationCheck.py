import torch.nn
import numpy as np

from hive.debugger.DebuggerInterface import DebuggerInterface
from hive.debugger.utils.metrics import almost_equal
from hive.debugger.utils import metrics
from hive.debugger.utils.model_params_getters import get_model_layer_names, get_model_weights_and_biases


class ActivationCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Activation"
        self.check_period = check_period
        self.iter_num = 0

    def set_output_metadata(self):
        self.outputs_metadata = {
            'non_zero_variance': {'patience': self.config["Output"]["patience"],
                                  'status': np.array([False])},
            'max_abs_greater_than_one': {'patience': self.config["Output"]["patience"],
                                         'status': np.array([False])},
            'can_be_negative': {'patience': self.config["Output"]["patience"],
                                'status': np.array([False])}
        }

    def update_outs_conds(self, outs_array):
        self.outputs_metadata['non_zero_variance']['status'] |= (np.var(outs_array, axis=0) > 0)
        self.outputs_metadata['max_abs_greater_than_one']['status'] |= (np.abs(outs_array) > 1).any(axis=0)
        self.outputs_metadata['can_be_negative']['status'] |= (outs_array < 0).any(axis=0)

    # def check_outputs(self, outs_array):
    #     if np.isinf(outs_array).any():
    #         self.react(self.main_msgs['out_inf']);
    #         return
    #     elif np.isnan(outs_array).any():
    #         self.react(self.main_msgs['out_nan']);
    #         return
    #     if (self.outputs_metadata['non_zero_variance']['status'] == False).any():
    #         self.outputs_metadata['non_zero_variance']['patience'] -= 1
    #         if self.outputs_metadata['non_zero_variance']['patience'] <= 0:
    #             self.react(self.main_msgs['out_cons'])
    #     else:
    #         self.outputs_metadata['non_zero_variance']['patience'] = self.config.out.patience
    #     if self.nn_data.model.problem_type == CLASSIFICATION_KEY:
    #         if outs_array.shape[1] == 1:
    #             positive = (outs_array >= 0.).all() and (outs_array <= 1.).all()
    #             if not (positive):
    #                 self.react(self.main_msgs['output_invalid'])
    #         else:
    #             # cannot check sum to 1.0 because of https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
    #             sum_to_one = (np.sum(outs_array, axis=1) > 0.95).all() and (np.sum(outs_array, axis=1) < 1.05).all()
    #             positive = (outs_array >= 0.).all()
    #             valid_n_outs = outs_array.shape[1] == self.targets_metadata['count']
    #             if not (positive and sum_to_one and valid_n_outs):
    #                 self.react(self.main_msgs['output_invalid'])
    #     elif self.nn_data.model.problem_type == REGRESSION_KEY:
    #         if len(outs_array.shape) > 1:
    #             valid_n_outs = outs_array.shape[1] == self.targets_metadata['count']
    #             if not (valid_n_outs):
    #                 self.react(self.main_msgs['output_invalid'])
    #         if (self.outputs_metadata['max_abs_greater_than_one']['status'] < self.targets_metadata[
    #             'max_abs_greater_than_one']).any():
    #             self.outputs_metadata['max_abs_greater_than_one']['patience'] -= 1
    #             if self.outputs_metadata['max_abs_greater_than_one']['patience'] <= 0:
    #                 self.react(self.main_msgs['lack_of_magnitude_express'])
    #         else:
    #             self.outputs_metadata['max_abs_greater_than_one']['patience'] = self.config.out.patience
    #         if (self.outputs_metadata['can_be_negative']['status'] < self.targets_metadata['can_be_negative']).any():
    #             self.outputs_metadata['can_be_negative']['patience'] -= 1
    #             if self.outputs_metadata['can_be_negative']['patience'] <= 0:
    #                 self.react(self.main_msgs['lack_of_negative_express'])
    #         else:
    #             self.outputs_metadata['can_be_negative']['patience'] = self.config.out.patience

    def run(self, predictions):
        if self.iter_num == 0:
            self.set_output_metadata()
        error_msg = list()
        self.iter_num += 1

        self.update_outs_conds(predictions)
        # if self.iter_num % self.config.period == 0:
        #     self.check_outputs(outputs)
        # acts = {k: (v, is_non_2d(v)) for k, v in acts.items()}
        # for acts_name, (acts_array, is_conv) in acts.items():
        #     acts_buffer = self.update_buffer(acts_name, acts_array)
        #     if self.iter_count < self.config.start or self.iter_count % self.config.period != 0: continue
        #     self.check_activations_range(acts_name, acts_buffer)
        #     if self.check_numerical_instabilities(acts_name, acts_array): continue
        #     if self.nn_data.model.act_fn_name in ['sigmoid', 'tanh']:
        #         self.check_saturated_layers(acts_name, acts_buffer, is_conv)
        #     else:
        #         self.check_dead_layers(acts_name, acts_buffer, is_conv)
        #     self.check_acts_distribution(acts_name, acts_buffer, is_conv)

        return error_msg
