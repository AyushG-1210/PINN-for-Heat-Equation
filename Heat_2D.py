import torch
import torch.nn as nn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

a = 2.1e-5  #diffusivity
T = 200.0   #time

#exact solution
def exact_solution(x, y, t):
    return torch.exp(-2 * (torch.pi**2) * a * t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

#fully connected network
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

#model definition
model = FCN(input=3, output=1, hidden=64, layers=4)

#collocation points
N_f = 2000
x_f = torch.rand(N_f, 1, requires_grad=True)
y_f = torch.rand(N_f, 1, requires_grad=True)
t_f = torch.rand(N_f, 1, requires_grad=True)
X_f = torch.cat([x_f, y_f, t_f], dim=1)

#initial condition (t=0)
N_ic = 200
x_ic = torch.rand(N_ic, 1)
y_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)
X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
u_ic = exact_solution(x_ic, y_ic, t_ic)

#boundary condition 
N_b = 200
t_b = torch.rand(N_b, 1)

# pick edges
edge_choice = torch.randint(low=0, high=4, size=(N_b,))
x_b = torch.zeros(N_b, 1)
y_b = torch.zeros(N_b, 1)
for i in range(N_b):
    if edge_choice[i] == 0:
        x_b[i] = 0.0; y_b[i] = torch.rand(1)
    elif edge_choice[i] == 1:
        x_b[i] = 1.0; y_b[i] = torch.rand(1)
    elif edge_choice[i] == 2:
        x_b[i] = torch.rand(1); y_b[i] = 0.0
    else:
        x_b[i] = torch.rand(1); y_b[i] = 1.0
X_b = torch.cat([x_b, y_b, t_b], dim=1)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

# training
n_epochs = 5000
print_interval = 1000

for it in range(n_epochs+1):
    optimizer.zero_grad()

    #ic loss
    u_ic_pred = model(X_ic)
    loss_ic = mse(u_ic_pred, u_ic)

    #bc loss
    u_b_pred = model(X_b)
    loss_bc = mse(u_b_pred, torch.zeros_like(u_b_pred))

    #physics loss
    u_pred = model(X_f)
    grads = torch.autograd.grad(u_pred, X_f, torch.ones_like(u_pred), create_graph=True)[0]
    u_x, u_y, u_t = grads[:,0:1], grads[:,1:2], grads[:,2:3]
    u_xx = torch.autograd.grad(u_x, X_f, torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    u_yy = torch.autograd.grad(u_y, X_f, torch.ones_like(u_y), create_graph=True)[0][:,1:2]
    residual = u_t - a*(u_xx + u_yy)
    loss_phys = mse(residual, torch.zeros_like(residual))

    #total loss
    loss = loss_ic + loss_bc + loss_phys
    loss.backward()
    optimizer.step()

    #plots
    if it % print_interval == 0:
        print(f"Iter {it}: Loss={loss.item():.4e}, IC={loss_ic.item():.3e}, BC={loss_bc.item():.3e}, Phys={loss_phys.item():.3e}")

        n_grid = 64
        x = torch.linspace(0,1,n_grid)
        y = torch.linspace(0,1,n_grid)
        Xg, Yg = torch.meshgrid(x,y, indexing='ij')

        times = [0.0, 0.25, 0.5, 1.0]
        plt.figure(figsize=(12,3))
        for j, t_val in enumerate(times,1):
            t_vis = torch.full_like(Xg, t_val)
            X_vis = torch.cat([Xg.reshape(-1,1), Yg.reshape(-1,1), t_vis.reshape(-1,1)], dim=1)
            u_pred = model(X_vis).detach().cpu().numpy().reshape(n_grid,n_grid)

            plt.subplot(1,len(times),j)
            plt.imshow(u_pred, extent=[0,1,0,1], origin='lower', cmap='hot', aspect='auto')
            plt.colorbar()
            plt.title(f"t={t_val}")
        plt.suptitle(f"Heat eqn PINN predictions, iter {it}")
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        t_val = times[-1]
        t_vis = torch.full_like(Xg, t_val)
        X_vis = torch.cat([Xg.reshape(-1,1), Yg.reshape(-1,1), t_vis.reshape(-1,1)], dim=1)
        u_pred = model(X_vis).detach().cpu().numpy().reshape(n_grid, n_grid)

        ax.plot_surface(Xg.numpy(), Yg.numpy(), u_pred, cmap='hot')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u(x,y,t)')
        ax.set_title(f'3D Surface at t={t_val}, iter {it}')
        plt.tight_layout()
        plt.show()
