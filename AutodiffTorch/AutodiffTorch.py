import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Parameter für den Balken
L = 2.0     # Länge des Balkens [m]
E = 210e9         # Elastizitätsmodul [Pa]
I = 1e-6          # Flächenträgheitsmoment [m^4]
q = 1000          # Streckenlast [N/m]

# Charakteristische Größe für dimensionslose Formulierung
w_char = q * L**4 / (E * I)

# Neuronales Netz
class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        layer_list = []
        for i in range(len(layers) - 1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                layer_list.append(nn.Tanh())
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)

# Generierung von Kollokationspunkten (in dimensionsloser Form: z ∈ [0,1])
def get_collocation_points(n_points):
    x = torch.linspace(0, 1, n_points, requires_grad=True).reshape(-1,1)
    return x

# Berechnung der analytischen Lösung (dimensionslos)
def analytical_solution(z):
    return (1/24) * z**2 * (z**2 - 4 * z + 6)

# Trainingsparameter
layers = [1, 20, 20, 20, 1]
net = DNN(layers)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

n_collocation = 100
x_coll = get_collocation_points(n_collocation)

n_epochs = 10000

for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Netzwerk-Ausgabe v(z)
    v = net(x_coll)

    # Berechnung der 4. Ableitung von v(z) nach z
    d1 = torch.autograd.grad(v, x_coll, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    d2 = torch.autograd.grad(d1, x_coll, grad_outputs=torch.ones_like(d1), create_graph=True)[0]
    d3 = torch.autograd.grad(d2, x_coll, grad_outputs=torch.ones_like(d2), create_graph=True)[0]
    d4 = torch.autograd.grad(d3, x_coll, grad_outputs=torch.ones_like(d3), create_graph=True)[0]

    # Residuum der dimensionslosen Differentialgleichung: d⁴v/dz⁴ = 1
    residual = d4 - 1.0
    loss_residual = loss_fn(residual, torch.zeros_like(residual))

    # Randbedingungen (alle in dimensionsloser Form z ∈ [0,1])
    z0 = torch.zeros(1,1, requires_grad=True)
    z1 = torch.ones(1,1, requires_grad=True)

    v0 = net(z0)
    dv0 = torch.autograd.grad(v0, z0, grad_outputs=torch.ones_like(v0), create_graph=True)[0]

    v1 = net(z1)
    dv1 = torch.autograd.grad(v1, z1, grad_outputs=torch.ones_like(v1), create_graph=True)[0]
    ddv1 = torch.autograd.grad(dv1, z1, grad_outputs=torch.ones_like(dv1), create_graph=True)[0]
    dddv1 = torch.autograd.grad(ddv1, z1, grad_outputs=torch.ones_like(ddv1), create_graph=True)[0]

    bc_loss = loss_fn(v0, torch.zeros_like(v0)) + \
              loss_fn(dv0, torch.zeros_like(dv0)) + \
              loss_fn(ddv1, torch.zeros_like(ddv1)) + \
              loss_fn(dddv1, torch.zeros_like(dddv1))

    loss = loss_residual + bc_loss
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Auswertung
z_eval = torch.linspace(0, 1, 100).reshape(-1,1)
with torch.no_grad():
    v_pred = net(z_eval).detach().numpy().flatten()
    v_exact = analytical_solution(z_eval).detach().numpy().flatten()

# Rücktransformation in dimensionalen Raum: w(x) = v(z) * w_char
x_eval = z_eval.numpy().flatten() * L
w_pred = v_pred * w_char
w_exact = v_exact * w_char

plt.figure(figsize=(8,5))
plt.plot(x_eval, w_exact, label='Analytisch')
plt.plot(x_eval, w_pred, '--', label='PINN')
plt.xlabel('x [m]')
plt.ylabel('w(x) [m]')
plt.legend()
plt.grid(True)
plt.title('Biegelinie eines Kragträgers unter gleichmäßiger Belastung')
plt.show()

# Relative Abweichung
rel_error = np.linalg.norm(w_exact - w_pred, 2) / np.linalg.norm(w_exact, 2)
print(f'Relative Fehlernorm: {rel_error:.2e}')
