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
num_epochs = 600
batch_size = 512
loss_hist = []
loss_data_hist = []
loss_phys_hist = []
loss_energy_hist = []
loss_theta_hist = []
loss_omega_hist = []
loss_alpha_hist = []
loss_phys_raw_hist = []