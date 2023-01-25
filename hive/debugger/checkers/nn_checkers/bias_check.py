import numpy as np

from hive.debugger.debugger_interface import DebuggerInterface
from hive.debugger.utils.model_params_getters import get_model_weights_and_biases


class BiasCheck(DebuggerInterface):

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "Bias"
        self.check_period = check_period

    def run(self, model):
        error_msg = list()
        _, initial_biases = get_model_weights_and_biases(model)
        if not initial_biases:
            error_msg.append(self.main_msgs['need_bias'])
        else:
            checks = []
            for b_name, b_array in initial_biases.items():
                checks.append(np.sum(b_array) == 0.0)
            if not np.all(checks):
                error_msg.append(self.main_msgs['zero_bias'])
        return error_msg


class OverfitBiasCheck(DebuggerInterface):

    # TODO: Fix the periodicity to make this check runs periodically
    # TODO: Fix the name of this check: find a better idea to group Bias Checks

    def __init__(self, check_period):
        super().__init__()
        self.check_type = "OverfitBias"
        self.check_period = check_period
        self.error_msg = list()

    def run(self, biases):
        for b_name, b_array in biases.items():
            if self.check_numerical_instabilities(b_name, b_array):
                continue

            b_reductions = np.mean(np.abs(b_array))
            self.check_divergence(b_name, b_reductions)

        return self.error_msg

    def check_numerical_instabilities(self, bias_name, bias_array):
        if self.config['numeric_ins']['disabled']:
            return
        if np.isinf(bias_array).any():
            self.error_msg.append(self.main_msgs['b_inf'].format(bias_name))
            return True
        if np.isnan(bias_array).any():
            self.error_msg.append(self.main_msgs['b_nan'].format(bias_name))
            return True
        return False

    def check_divergence(self, bias_name, bias_reductions):
        if self.config['div']['disabled']:
            return
        if bias_reductions[-1] > self.config['div']['mav_max_thresh']:
            self.error_msg.append(self.main_msgs['b_div_1'].format(bias_name, bias_reductions[-1],
                                                                   self.config.div.mav_max_thresh))
        elif len(bias_reductions) >= self.config['div']['window_size']:
            inc_rates = np.array(
                [bias_reductions[-i] / bias_reductions[-i - 1] for i in range(1, self.config['div']['window_size'])])
            if (inc_rates >= self.config['div']['inc_rate_max_thresh']).all():
                self.error_msg.append(self.main_msgs['b_div_2'].format(bias_name, max(inc_rates),
                                                                       self.config['div']['inc_rate_max_thresh']))

    # def begin(self):
    #     self.bias_tensors = self.nn_data.model.biases
    #     self.iter_count = -1
    #
    # def before_run(self, run_context):
    #     return SessionRunArgs(self.bias_tensors)
    #
    #
    # def update_buffer(self, bias_name, bias_array):
    #     self.nn_data.biases_reductions[bias_name].append(np.mean(np.abs(bias_array)))
    #     return self.nn_data.biases_reductions[bias_name]
    #
    # def after_run(self, run_context, run_values):
    #     self.iter_count += 1
    #     self.biases = run_values.results
    #     if not (self.biases): return  # no bias
    #     for b_name, b_array in self.biases.items():
    #         if self.check_numerical_instabilities(b_name, b_array):
    #             continue
    #         buffer_values = self.update_buffer(b_name, b_array)
    #         if self.iter_count < self.config.start or self.iter_count % self.config.period != 0:
    #             continue
    #         self.check_divergence(b_name, buffer_values)
