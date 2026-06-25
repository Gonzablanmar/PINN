import numpy as np
import torch
import torch.nn as nn


class TanhSin(nn.Module):
    def forward(self, x):
        return torch.tanh(x) + torch.sin(x)
    

# PHYSICAL INFORMED NEURAL NETWORK
class PINN_DoublePendulum(nn.Module):
    def __init__(self, input_dim=1, output_dim=2, hidden_dim=284, num_layers=4):
        super().__init__()
        
        layers = []
        
        # TURN t INTO AN INTERN VECTOR
        layers.append(nn.Linear(input_dim, hidden_dim))
        # ACTIVATION FUNCTION
        layers.append(nn.Tanh())
        
        # HIDDEN LAYERS
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # OUTPUT -> [THETA1, THETA2]
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    # FORWARD; THETA(t) = NN(t)
    def forward(self, t):
        return self.model(t)

def time_derivatives(model, t):
    t.requires_grad_(True)

    theta = model(t)

    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]

    dtheta1 = torch.autograd.grad(
        theta1, t,
        grad_outputs=torch.ones_like(theta1),
        create_graph=True, retain_graph=True
    )[0]

    dtheta2 = torch.autograd.grad(
        theta2, t,
        grad_outputs=torch.ones_like(theta2),
        create_graph=True, retain_graph=True
    )[0]

    dtheta = torch.hstack((dtheta1, dtheta2))

    # Segunda derivada
    d2theta1 = torch.autograd.grad(
        dtheta1, t,
        grad_outputs=torch.ones_like(dtheta1),
        create_graph=True
    )[0]

    d2theta2 = torch.autograd.grad(
        dtheta2, t,
        grad_outputs=torch.ones_like(dtheta2),
        create_graph=True
    )[0]

    d2theta = torch.hstack((d2theta1, d2theta2))

    return theta, dtheta, d2theta

def compute_loss(model, t_norm, theta_norm, omega_norm, alpha_norm, t_min, t_max):
    
    theta_pred, dtheta_pred, d2theta_pred = time_derivatives(model, t_norm)
    
    dt_scale = (t_max - t_min)
    
    dtheta_pred_phys = dtheta_pred / dt_scale
    d2theta_pred_phys = d2theta_pred / (dt_scale**2)
    
    loss_theta = torch.mean((theta_pred - theta_norm)**2)
   
    
    loss = loss_theta 
    
    return loss, loss_theta

def double_pendulum_acc(theta, omega, params):
    
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]
    
    omega1 = omega[:, 0:1]
    omega2 = omega[:, 1:2]
    
    m1, m2, l1, l2, g = params
    
    delta = theta1 - theta2
    
    denom1 = l1 * (2*m1 + m2 - m2 * torch.cos(2*delta)) + 1e-6
    denom2 = l2 * (2*m1 + m2 - m2 * torch.cos(2*delta)) + 1e-6
    
    # alpha1
    num1 = (
        -g * (2*m1 + m2) * torch.sin(theta1)
        - m2 * g * torch.sin(theta1 - 2*theta2)
        - 2 * torch.sin(delta) * m2 * (
            omega2**2 * l2 + omega1**2 * l1 * torch.cos(delta)
        )
    )
    
    alpha1 = num1 / denom1
    
    # alpha2
    num2 = (
        2 * torch.sin(delta) * (
            omega1**2 * l1 * (m1 + m2)
            + g * (m1 + m2) * torch.cos(theta1)
            + omega2**2 * l2 * m2 * torch.cos(delta)
        )
    )
    
    alpha2 = num2 / denom2
    
    return torch.hstack((alpha1, alpha2))

def physical_loss(theta_pred, dtheta_pred, d2theta_pred,
                  t_min, t_max, params,
                  theta_mean, theta_std,
                  omega_mean, omega_std,
                  alpha_mean, alpha_std):

    # =========================
    # Escalado temporal
    # =========================
    dt_scale = (t_max - t_min)

    omega_pred = dtheta_pred / dt_scale
    alpha_pred = d2theta_pred / (dt_scale**2)

    # =========================
    # DESNORMALIZACIÓN
    # =========================
    theta_real = theta_pred * theta_std + theta_mean
    omega_real = omega_pred * omega_std + omega_mean
    alpha_real = alpha_pred 

    # =========================
    # MODELO FÍSICO
    # =========================
    alpha_phys = double_pendulum_acc(theta_real, omega_real, params)

    # =========================
    # RESIDUAL (CLAVE)
    # =========================
    residual = alpha_real - alpha_phys

    # 🔥 CLIPPING SUAVE DEL RESIDUAL (MUY IMPORTANTE)
    residual = torch.clamp(residual, -30, 30)

    # =========================
    # LOSS FÍSICO
    # =========================
    loss_phys = torch.mean(residual**2)

    # =========================
    # PROTECCIÓN ANTI-NaN
    # =========================
    if torch.isnan(loss_phys):
        print("NaN en loss_phys")
        return torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

    return loss_phys
def energy(theta, omega, params): 
    theta1 = theta[:, 0:1]
    theta2 = theta[:, 1:2]
    
    omega1 = omega[:, 0:1]
    omega2 = omega[:, 1:2]
    
    m1, m2, l1, l2, g = params
    
    # Energía cinética
    T = 0.5 * m1 * (l1 * omega1)**2 + \
        0.5 * m2 * (
            (l1 * omega1)**2 +
            (l2 * omega2)**2 +
            2 * l1 * l2 * omega1 * omega2 * torch.cos(theta1 - theta2)
        )
    
    # Energía potencial
    V = -(m1 + m2) * g * l1 * torch.cos(theta1) \
        - m2 * g * l2 * torch.cos(theta2)
    
    return T + V
    


def energy_loss(theta_real, omega_real, params):
    
    E = energy(theta_real, omega_real, params)
    
    # comparar con valor inicial
    E0 = torch.mean(E)
    
    loss_E = torch.mean((E - E0)**2)
    
    return loss_E