"""
                    0 C
        ___________________________
       |                           |
       |                           |
       |                           |
       |                           |
       |                           |
 100 C |            0 C            | insulated
       |                           |
       |                           |
       |                           |
       |                           |
       |___________________________|
                    0 C
T(x,y,0) = 0
T(0,y,t) = 100
dT(L,y,t)/dx = 0 (Boundary treatment for FVM required)
T(x,0,t) = 0
T(x,H,t) = 0

Assumptions
1. unsteady heat conduction
2. no heat generation
3. uniform thermal properties
4. Square plate
5. dx = dy, Lx = Ly

Fo = alpha*dt/dx^2
"""

import numpy as np
from matplotlib import cm, pyplot as plt

L = 1
nx = 120
dx = L/nx
Fo = 0.25
# alpha = 1
# dt = Fo*dx**2/alpha
nit = 1000000

def apply_boundary(T):
    T[-1, :] = 90
    T[:, 0] = 0
    T[:, -1] = 0
    T[0, :] = T[1, :]

def convergence_criteria(T, T_old):
    return np.max(np.abs(T - T_old))

def explicit_soln():
    T = np.zeros((nx, nx))
    T_old = T.copy()
    for n in range(nit):
        apply_boundary(T)

        T[1:-1,1:-1] = T_old[1:-1,1:-1] + Fo*(T_old[0:-2,1:-1] + T_old[2:,1:-1] + T_old[1:-1,0:-2] + T_old[1:-1,2:] - 4*T_old[1:-1,1:-1])

        if convergence_criteria(T, T_old) < 1e-6:
            print(f"Explicit converged at n={n}")
            break

        T_old = T.copy()
    return T, n

def implicit_soln():
    T = np.zeros((nx, nx))
    T_old = T.copy()
    for n in range(nit):
        apply_boundary(T)

        T[1:-1,1:-1] = (T_old[1:-1,1:-1] + Fo*(T[0:-2,1:-1] + T[2:,1:-1] + T[1:-1,0:-2] + T[1:-1,2:]))/(1 + 4*Fo)

        if convergence_criteria(T, T_old) < 1e-6:
            print(f"Implicit converged at n={n}")
            break

        T_old = T.copy()
    return T, n

exp_T, exp_n = explicit_soln()
imp_T, imp_n = implicit_soln()

x = np.linspace(0, L, nx)
y = np.linspace(0, L, nx)
X,Y = np.meshgrid(x, y)
plt.subplots(1, 2)
plt.subplot(1, 2, 1)
plt.contourf(X, Y, exp_T, cmap=cm.seismic)
plt.colorbar()
plt.title(f'Explicit Solution (converged at n={exp_n})')
plt.xlabel('X')
plt.ylabel('Y')
plt.subplot(1, 2, 2)
plt.contourf(X, Y, imp_T, cmap=cm.seismic)
plt.colorbar()
plt.title(f'Implicit Solution (converged at n={imp_n})')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
