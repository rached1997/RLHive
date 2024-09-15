from hive.debugger.checkers.nn_checkers.activationCheck import ActivationCheck
from hive.debugger.checkers.nn_checkers.bias_check import BiasCheck, OverfitBiasCheck
from hive.debugger.checkers.nn_checkers.loss_check import LossCheck, OverfitLossCheck
from hive.debugger.checkers.nn_checkers.observations_check import ObservationsCheck
from hive.debugger.checkers.nn_checkers.proper_fitting_check import ProperFittingCheck
from hive.debugger.checkers.nn_checkers.gradient_check import GradientCheck
from hive.debugger.checkers.nn_checkers.weights_check import WeightsCheck, OverfitWeightsCheck
from hive.debugger.debugger_interface import DebuggerInterface
from hive.utils.registry import registry

# Todo
#  registry.register_all(
#     Debugger,
#     {
#         "NullDebugger": NullDebugger,
#         "PreCheckDebugger": PreCheckDebugger,
#         "PostCheckDebugger": PostCheckDebugger,
#         "OnTrainingCheckDebugger": OnTrainingCheckDebugger,
#         "CompositeDebugger": CompositeDebugger,
#     },
# )

registry.register("Observation", ObservationsCheck, ObservationsCheck)
registry.register("Weight", WeightsCheck, WeightsCheck)
registry.register("Bias", BiasCheck, BiasCheck)
registry.register("Loss", LossCheck, LossCheck)
registry.register("ProperFitting", ProperFittingCheck, ProperFittingCheck)
registry.register("Activation", ActivationCheck, ActivationCheck)
registry.register("Gradient", GradientCheck, GradientCheck)
registry.register("OverfitBias", OverfitBiasCheck, OverfitBiasCheck)
registry.register("OverfitWeight", OverfitWeightsCheck, OverfitWeightsCheck)
registry.register("OverfitLoss", OverfitLossCheck, OverfitLossCheck)


get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")
