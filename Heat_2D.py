#imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

#setup
def exact_solution(x, y, t):
    a = 0.01
    exp = -2 * np.pi**2 * a * t
    return torch.exp(torch.tensor(exp)) * torch.sin(np.pi * x) * torch.sin(np.pi * y)

class FCN(nn.Module):
    def __init__(self, input, output, hidden, layers):
        super().__init__()
        activation = nn.Tanh

        self.pinn = nn.Sequential(
            nn.Linear(input, hidden),
            activation(),
            *[nn.Sequential(nn.Linear(hidden, hidden), activation()) for _ in range(layers)],
            nn.Linear(hidden, output)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.pinn(x)

#definition
model = FCN(input=3, output=1, hidden=64, layers=4)

#collocation points
N_f = 1000
x_f = torch.rand(N_f, 1, requires_grad=True)
y_f = torch.rand(N_f, 1, requires_grad=True)
t_f = torch.rand(N_f, 1, requires_grad=True)
X_f = torch.cat([x_f, y_f, t_f], dim=1)

#boundary conditions
N_b = 100
x_b = torch.rand(N_b, 1, requires_grad=True)
y_b = torch.rand(N_b, 1, requires_grad=True)
t_b = torch.zeros(N_b, 1, requires_grad=True)
X_b = torch.cat([x_b, y_b, t_b], dim=1)

#training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

a = 0.01
for i in range(10000):
    optimizer.zero_grad()

    #boundary loss (initial condition at t=0)
    u_b = model(X_b)
    u_b_exact = exact_solution(x_b, y_b, t_b)
    loss_boundary = nn.MSELoss()(u_b, u_b_exact)

    #physics loss
    u = model(X_f)

    grads = torch.autograd.grad(u, X_f, torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_y = grads[:, 1:2]
    u_t = grads[:, 2:3]

    u_xx = torch.autograd.grad(u_x, X_f, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X_f, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

    residual = u_t - a * (u_xx + u_yy)
    loss_physics = nn.MSELoss()(residual, torch.zeros_like(residual))

    #total loss
    loss = loss_boundary + loss_physics
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f"Iter {i}, Loss: {loss.item():.6f}, Boundary: {loss_boundary.item():.6f}, Physics: {loss_physics.item():.6f}")

        #heatmap visualization
        n_grid = 50
        x_grid = torch.linspace(0, 1, n_grid)
        y_grid = torch.linspace(0, 1, n_grid)
        X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
        t_vis = torch.full_like(X, 0.5)
        X_vis = torch.cat([X.reshape(-1,1), Y.reshape(-1,1), t_vis.reshape(-1,1)], dim=1)
        u_pred = model(X_vis).detach().cpu().numpy().reshape(n_grid, n_grid)

        plt.figure(figsize=(5,4))
        plt.imshow(u_pred, extent=[0,1,0,1], origin='lower', aspect='auto', cmap='hot')
        plt.colorbar(label='u(x, y, t=0.5)')
        plt.title(f'Predicted Heatmap at t=0.5, Iter {i}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.show()