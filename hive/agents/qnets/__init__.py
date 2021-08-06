from hive import registry
from hive.agents.qnets.atari import NatureAtariDQNModel
from hive.agents.qnets.base import FunctionApproximator
from hive.agents.qnets.conv import SimpleConvModel
from hive.agents.qnets.mlp import SimpleMLP
from hive.agents.qnets.rainbow_mlp import ComplexMLP, DistributionalMLP
from hive.agents.qnets.rainbow_conv import ComplexConv, DistributionalConv

registry.register_all(
    FunctionApproximator,
    {
        "SimpleMLP": FunctionApproximator(SimpleMLP),
        "ComplexMLP": FunctionApproximator(ComplexMLP),
        "DistributionalMLP": FunctionApproximator(DistributionalMLP),
        "SimpleConvModel": FunctionApproximator(SimpleConvModel),
        "ComplexConv": FunctionApproximator(ComplexConv),
        "DistributionalConv": FunctionApproximator(DistributionalConv),
        "NatureAtariDQNModel": FunctionApproximator(NatureAtariDQNModel),
    },
)

get_qnet = getattr(registry, f"get_{FunctionApproximator.type_name()}")
