import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Material- und Balkenparameter
E = 210e9      # Elastizitätsmodul [Pa] (z.B. Stahl)
I = 1e-6       # Flächenträgheitsmoment [m^4]
EI = E * I

L = 1.0        # Balkenlänge [m]
q0 = 1000.0    # konstante Streckenlast [N/m]

device = torch.device("cpu")  # ggf. "cuda"

# Streckenlast-Funktion q(x)
def q(x):
    return q0 * torch.ones_like(x)

# PINN-Modell
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1)
        )
    
    def forward(self, x):
        return self.hidden(x)

model = Net().to(device)

# Punkte zur Approximation
n_collocation = 100
x = torch.linspace(0, L, n_collocation, requires_grad=True).view(-1,1).to(device)

# Berechnung der vierten Ableitung mittels Autograd
def fourth_derivative(model, x):
    w = model(x)
    dw = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    d2w = torch.autograd.grad(dw, x, grad_outputs=torch.ones_like(dw), create_graph=True)[0]
    d3w = torch.autograd.grad(d2w, x, grad_outputs=torch.ones_like(d2w), create_graph=True)[0]
    d4w = torch.autograd.grad(d3w, x, grad_outputs=torch.ones_like(d3w), create_graph=True)[0]
    return d4w

# Loss-Funktion (PDE + Randbedingungen)
def loss_fn():
    d4w = fourth_derivative(model, x)
    residual = EI * d4w - q(x)
    pde_loss = torch.mean(residual**2)

    # Randbedingungen (x=0 fest eingespannt → w(0)=0, w'(0)=0)
    x0 = torch.tensor([[0.0]], requires_grad=True).to(device)
    w0 = model(x0)
    dw0 = torch.autograd.grad(w0, x0, grad_outputs=torch.ones_like(w0), create_graph=True)[0]

    bc_loss = w0.pow(2).mean() + dw0.pow(2).mean()
    
    return pde_loss + bc_loss

# Training
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 5000
for epoch in range(n_epochs):
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.3e}")

# Vorhersage und Visualisierung
x_test = torch.linspace(0, L, 200).view(-1,1).to(device)
w_pred = model(x_test).detach().cpu().numpy()

plt.figure(figsize=(8,4))
plt.plot(x_test.cpu(), w_pred*1e3, label="PINN (w) [mm]")
plt.xlabel("x [m]")
plt.ylabel("w(x) [mm]")
plt.title("Biegelinie eines eingespannten Balkens (PINN)")
plt.grid(True)
plt.legend()
plt.show()

