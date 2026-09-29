import numpy as np
import torch
import torch.nn as nn


class TanhSin(nn.Module):
    def forward(self, x):
        return torch.tanh(x) + torch.sin(x)
    

class PINN_DoublePendulum(nn.Module):
    def __init__(
        self,
        theta_std,
        omega_mean,
        omega_std,
        dt_scale,
        hidden_size=256
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),

            nn.Linear(hidden_size, 2)
        )

        # Se guardan dentro del modelo para convertir la
        # velocidad inicial a la escala normalizada del ángulo
        self.register_buffer(
            "theta_std",
            theta_std.detach().clone().float()
        )

        self.register_buffer(
            "omega_mean",
            omega_mean.detach().clone().float()
        )

        self.register_buffer(
            "omega_std",
            omega_std.detach().clone().float()
        )

        self.dt_scale = float(dt_scale)

    def forward(self, x):
        """
        Entrada:
        x[:, 0]   = tiempo normalizado
        x[:, 1:3] = theta inicial normalizada
        x[:, 3:5] = omega inicial normalizada

        Salida:
        theta normalizada
        """

        t_norm = x[:, 0:1]
        theta0_norm = x[:, 1:3]
        omega0_norm = x[:, 3:5]

        # Recuperar velocidad inicial física, rad/s
        omega0_real = (
            omega0_norm * self.omega_std
            + self.omega_mean
        )

        # Convertir la velocidad física a:
        # d(theta_norm) / d(t_norm)
        initial_slope_norm = (
            omega0_real
            * self.dt_scale
            / self.theta_std
        )

        # Corrección libre aprendida por la red
        correction = self.net(x)

        # Condiciones iniciales impuestas exactamente
        theta_norm = (
            theta0_norm
            + t_norm * initial_slope_norm
            + t_norm.pow(2) * correction
        )

        return theta_norm
def time_derivatives(model, t, initial_conditions):
    """
    t:
        Tiempo normalizado, shape [batch, 1].

    initial_conditions:
        Condiciones iniciales normalizadas, shape [batch, 4]
        [theta1_0, theta2_0, omega1_0, omega2_0].
    """

    # Creamos un tensor independiente para poder derivar respecto al tiempo
    if not t.requires_grad:
        t = t.clone().detach().requires_grad_(True)

    # Entrada completa de la PINN
    model_input = torch.cat(
        [t, initial_conditions],
        dim=1
    )

    # Ángulos normalizados
    theta = model(model_input)

    first_derivatives = []
    second_derivatives = []

    for output_index in range(2):
        theta_i = theta[:, output_index:output_index + 1]

        # Primera derivada respecto al tiempo
        dtheta_i = torch.autograd.grad(
            outputs=theta_i,
            inputs=t,
            grad_outputs=torch.ones_like(theta_i),
            create_graph=True,
            retain_graph=True
        )[0]

        # Segunda derivada respecto al tiempo
        d2theta_i = torch.autograd.grad(
            outputs=dtheta_i,
            inputs=t,
            grad_outputs=torch.ones_like(dtheta_i),
            create_graph=True,
            retain_graph=True
        )[0]

        first_derivatives.append(dtheta_i)
        second_derivatives.append(d2theta_i)

    dtheta = torch.cat(first_derivatives, dim=1)
    d2theta = torch.cat(second_derivatives, dim=1)

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

    dt_scale = (t_max - t_min)

    # Variables físicas
    theta_real = theta_pred * theta_std + theta_mean

    omega_real = (
        dtheta_pred * theta_std
    ) / dt_scale

    alpha_real = (
        d2theta_pred * theta_std
    ) / (dt_scale**2)

    alpha_phys = double_pendulum_acc(
        theta_real,
        omega_real,
        params
    )

    residual = alpha_real - alpha_phys

    loss_phys = torch.mean(
        torch.log1p(residual**2)
    )

    if torch.isnan(loss_phys):
        return torch.tensor(
            0.0,
            dtype=torch.float32,
            requires_grad=True
        )

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
    E0 = E[0:1]
    
    loss_E = torch.mean((E - E0)**2)
    
    return loss_E