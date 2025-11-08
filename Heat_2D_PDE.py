import torch
import torch.nn as nn
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

#non-dimensional parameters
a = 1.0      #non-dimensional diffusivity
T = 1.0      #time

#physical parameters
L = 1.0
a_phys = 2.1e-5  #physical diffusivity

#exact solution
def exact_solution(x, y, t):
    return torch.exp(-2 * (torch.pi**2) * a * t) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

#fully connected network
class FCN(nn.Module):
    def __init__(self, input, output, hidden, layers):
        super().__init__()
        activation = nn.SiLU
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

#move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#collocation points
N_f = 8000
x_f = torch.rand(N_f, 1, requires_grad=True)
y_f = torch.rand(N_f, 1, requires_grad=True)
t_f = torch.rand(N_f, 1, requires_grad=True)
X_f = torch.cat([x_f, y_f, t_f], dim=1)

#initial condition (t=0)
N_ic = 1000
x_ic = torch.rand(N_ic, 1)
y_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)
X_ic = torch.cat([x_ic, y_ic, t_ic], dim=1)
u_ic = exact_solution(x_ic, y_ic, t_ic)

#boundary condition
N_b = 200
t_b = torch.rand(N_b, 1)

#pick edges
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

#move tensors to device
X_f, X_ic, X_b = X_f.to(device), X_ic.to(device), X_b.to(device)
u_ic = u_ic.to(device)

#training
epochs = 15000
print_interval = 1000
check_plateau = False
loss_history = []

#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
mse = nn.MSELoss()

#scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=320,
    # verbose=True # Removed verbose argument
)

loss_history = []

for i in range(epochs+1):
    optimizer.zero_grad()

    #initial condition loss
    u_ic_pred = model(X_ic)
    loss_ic = mse(u_ic_pred, u_ic)

    #boundary condition loss
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
    lam_bc = 1.0
    lam_ic = 3.0
    lam_phys = 12.0
    loss = loss_ic*lam_ic + loss_bc*lam_bc + loss_phys*lam_phys
    loss.backward()
    optimizer.step()

    #scheduler step
    scheduler.step(loss.item())
    loss_history.append(loss.item())

    #plots
    if i % print_interval == 0:
        print(f"Iter {i}: Loss={loss.item():.4e}, IC={loss_ic.item():.3e}, BC={loss_bc.item():.3e}, Phys={loss_phys.item():.3e}")

        grid = 64
        x = torch.linspace(0,1,grid)
        y = torch.linspace(0,1,grid)
        Xg, Yg = torch.meshgrid(x,y, indexing='ij')

        times_3d = [0.0, 0.25, 0.5, 0.75, 1.0]
        errors = []

        #3D surface plots over time
        fig = plt.figure(figsize=(20,4))
        for idx, t_val in enumerate(times_3d, 1):
            ax = fig.add_subplot(1, 5, idx, projection='3d')
            t_vis = torch.full_like(Xg, t_val)
            X_vis = torch.cat([Xg.reshape(-1,1), Yg.reshape(-1,1), t_vis.reshape(-1,1)], dim=1)
            X_vis = X_vis.to(device) # Move X_vis to the correct device
            u_pred = model(X_vis).detach().cpu().numpy().reshape(grid, grid)
            t_phys = t_val * L**2 / a_phys
            ax.plot_surface(Xg.numpy(), Yg.numpy(), u_pred, cmap='hot')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u(x,y,t)')
            ax.set_title(f't={t_val} (phys={t_phys:.2f}s)')

            #compute relative L2 error and store
            u_exact = exact_solution(Xg.reshape(-1,1), Yg.reshape(-1,1), t_vis.reshape(-1,1)).cpu().numpy().reshape(grid, grid)
            rel_l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
            errors.append(rel_l2_error)

        plt.suptitle(f'Heat eqn PINN predictions, iter {i}')
        plt.tight_layout()
        plt.show()

        # Plot relative L2 errors over time
        plt.figure(figsize=(7,4))
        plt.plot(times_3d, errors, marker='o')
        plt.xlabel('Time (non-dimensional)')
        plt.ylabel('Relative L2 Error')
        plt.title(f'Relative L2 Error vs. Time, iter {i}')
        plt.yscale('log') # Use a log scale for better visualization of errors
        plt.tight_layout()
        plt.show()


#final loss
plt.figure(figsize=(7,4))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.title('Loss Curve During Training')
plt.yscale('log')
plt.tight_layout()
plt.show()

#save model
torch.save(model.state_dict(), 'model.pt')
