import numpy as np  
from typing import TypeAlias
from .backends.backend_np import Device,cpu

NDArray:TypeAlias = np.ndarray
import numpy as array_api

class Value:
    pass

class Op:
    pass

class Tensor:
    def __init__(self,array:NDArray,device:Device=cpu(),dtype:str='float32',requires_grad:bool=False):
       pass 
   
