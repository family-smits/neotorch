"""NeoTorch — lokale tensor-runtime voor 3D-generatie (geen API-keys, geen Ollama)."""

from .tensor import Tensor, zeros, ones, randn, from_numpy
from .nn import Module, Linear, Sequential, ReLU, Tanh, Sigmoid
from .mesh import Mesh
from .generate import Model3DGenerator
from .export import export_obj, export_stl
from .mesh import PRACTICAL_MAX_VERTS, THEORETICAL_MAX_VERTS
from .tokens import TokenWallet, DAILY_LIMIT, SHAPE_COST

__version__ = "0.2.0"
__all__ = [
    "Tensor",
    "zeros",
    "ones",
    "randn",
    "from_numpy",
    "Module",
    "Linear",
    "Sequential",
    "ReLU",
    "Tanh",
    "Sigmoid",
    "Mesh",
    "Model3DGenerator",
    "export_obj",
    "export_stl",
    "TokenWallet",
    "DAILY_LIMIT",
    "SHAPE_COST",
    "PRACTICAL_MAX_VERTS",
    "THEORETICAL_MAX_VERTS",
]
