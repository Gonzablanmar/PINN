import numpy as np
import pandas as pd
dt = 0.01
params = (
    1.0,  # m1
    1.0,  # m2
    1.0,  # l1
    1.0,  # l2
    9.81  # g
)
num_epochs = 3000
batch_size = 512