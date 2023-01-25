from hive.debugger.Checkers.NN_checkers.ActivationCheck import ActivationCheck
from hive.debugger.Checkers.NN_checkers.BiasCheck import BiasCheck
from hive.debugger.Checkers.NN_checkers.LossCheck import LossCheck
from hive.debugger.Checkers.NN_checkers.ObservationsCheck import ObservationsCheck
from hive.debugger.Checkers.NN_checkers.ProperFittingCheck import ProperFittingCheck
from hive.debugger.Checkers.NN_checkers.WeightsCheck import WeightsCheck
from hive.debugger.DebuggerInterface import DebuggerInterface
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



get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")
