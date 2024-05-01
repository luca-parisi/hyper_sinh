"""
This file contains constants leveraged by the hyper-sinh activation function.
"""

import torch

GT_COEFFICIENT = 1/3
LE_FIRST_COEFFICIENT = 1/4
LE_SECOND_COEFFICIENT = 3

GT_COEFFICIENT_PYTORCH = torch.tensor(GT_COEFFICIENT)
LE_FIRST_COEFFICIENT_PYTORCH = torch.tensor(LE_FIRST_COEFFICIENT)
LE_SECOND_COEFFICIENT_PYTORCH = torch.tensor(LE_SECOND_COEFFICIENT)

NAME_HYPER_SINH = 'hyper_sinh'
