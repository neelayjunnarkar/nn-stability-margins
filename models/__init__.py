"""
This folder contains controller models and an implicit model for system identification.
"""


# from models.ProjRNNOld import ProjRNNOldModel
# from models.ProjRNN import ProjRNNModel
from models.dissipative_RINN import DissipativeRINN
from models.dissipative_simplest_RINN import DissipativeSimplestRINN
from models.dissipative_theta_RINN import DissipativeThetaRINN
from models.fully_connected import FullyConnectedNetwork
from models.implicit_model import ImplicitModel
from models.LTI import LTIModel
from models.RINN import RINN
from models.RNN import RNN

model_map = {
    "<class 'models.implicit_model.ImplicitModel'>": ImplicitModel,
    "<class 'models.dissipative_simplest_RINN.DissipativeSimplestRINN'>": DissipativeSimplestRINN,
    "<class 'models.LTI.LTIModel'>": LTIModel,
    "<class 'models.RINN.RINN'>": RINN,
}
