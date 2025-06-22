
import numpy as np
import matplotlib.pyplot as plt

# Balkenparameter
L = 1.0           # Länge des Balkens [m]
E = 210e9         # Elastizitätsmodul [Pa] (z.B. Stahl)
I = 1e-6          # Flächenträgheitsmoment [m^4]
q = 1000          # Streckenlast [N/m]

# Finite-Elemente-Parameter
n_elem = 10                  # Anzahl Elemente
n_nodes = n_elem + 1         # Anzahl Knoten
x = np.linspace(0, L, n_nodes)  # Knotenkoordinaten

# Elementsteifigkeitsmatrix für Euler-Bernoulli-Balken (2 Knoten, 2 Freiheitsgrade pro Knoten)
def element_stiffness(E, I, l):
    return (E * I / l**3) * np.array([
        [12, 6*l, -12, 6*l],
        [6*l, 4*l**2, -6*l, 2*l**2],
        [-12, -6*l, 12, -6*l],
        [6*l, 2*l**2, -6*l, 4*l**2]
    ])

# Elementlastvektor (gleichmäßig verteilte Last)
def element_load(q, l):
    return q * l / 2 * np.array([1, l/6, 1, -l/6])

# Gesamte Systemsteifigkeitsmatrix und Lastvektor aufbauen
K = np.zeros((2*n_nodes, 2*n_nodes))
f = np.zeros(2*n_nodes)

l_e = L / n_elem  # Elementlänge

for e in range(n_elem):
    ke = element_stiffness(E, I, l_e)
    fe = element_load(q, l_e)
    
    dofs = np.array([2*e, 2*e+1, 2*e+2, 2*e+3])
    
    K[np.ix_(dofs, dofs)] += ke
    f[dofs] += fe

np.set_printoptions(precision=3, suppress=False)
print(K)

# Randbedingungen: Fest eingespannt bei x=0 → Verschiebung und Rotation = 0
fixed_dofs = [0, 1]
free_dofs = np.setdiff1d(np.arange(2*n_nodes), fixed_dofs)

# Gleichungssystem lösen
K_ff = K[np.ix_(free_dofs, free_dofs)]
f_f = f[free_dofs]

print("K_ff: ")
print(K_ff)

print("f_f: ")
print(f_f)

u_f = np.linalg.solve(K_ff, f_f)

# Gesamtlösung zusammensetzen
u = np.zeros(2*n_nodes)
u[free_dofs] = u_f

# Ausgabe: Vertikale Verschiebung (jede zweite Komponente)
w = u[0::2]           # Vertikale Verschiebungen
theta = u[1::2]       # Rotationen (Biegewinkel in rad)

print("w:")
print(w)

np.set_printoptions(precision=10, suppress=False)
print("theta:")
print(theta)

# Plot der Biegelinie
plt.figure(figsize=(8, 4))
plt.plot(x, w*1e3, marker='o')
plt.xlabel("x [m]")
plt.ylabel("w(x) [mm]")
plt.title("Biegelinie eines eingespannten Balkens unter Streckenlast")
plt.grid(True)
plt.show()

# Plot der Rotationen (Biegewinkel)
plt.figure(figsize=(8, 4))
plt.plot(x, theta, marker='o')
plt.xlabel("x [m]")
plt.ylabel("θ(x) [rad]")  # ✔️ Jetzt korrekt beschriftet
plt.title("Biegewinkel eines eingespannten Balkens unter Streckenlast")
plt.grid(True)
plt.show()
