"""
# A good resource to refer: https://www.scribd.com/presentation/428957927/Couette-Flow-Solution-Using-Analytic-and-Crank-nicolson-Numerical

Assumptions of Couette flow:
1. The flow is unsteady incompressible.
2. The flow is laminar and fully developed. (v=0)
3. No pressure gradient in the flow direction. (dp/dx=0)
4. Developed flow has no change in velocity in x direction. (du/dx=0)
5. Uniform properties (nu, rho)

Governing equation: du/dt = nu * d^2u/dy^2
Non dimensionalized: du*/dt* = (1/Re)*(TU/H) * d^2u*/dy*^2

Solved using FVM

No separate boundary treatment required for Dirichlet conditions
"""
import numpy as np
from matplotlib import pyplot as plt



ny = 100
H = 1
U = 1
T = 1
Re = 200
dy = H / ny
nit = 10000000
F = 0.5 # something like a CFL number
dt = F * dy**2 * Re / (T*U)

def apply_boundary(u):
    u[0] = 0
    u[-1] = U

# explicit solution
# initialization
u = np.zeros(ny)
u_old = u.copy()

def convergence_criteria(u, u_old):
    return np.max(np.abs(u - u_old))

for n in range(nit):
    apply_boundary(u)

    d2u_dy2 = (u_old[2:] + u_old[0:-2] - 2*u_old[1:-1])/(dy**2)
    non_dimensional = (1/Re)*(T*U/H)

    u[1:-1] = u_old[1:-1] + dt * (non_dimensional * d2u_dy2)

    if convergence_criteria(u, u_old) < 1e-6:
        print(f'Explicit converged after {n} iterations, dt = {dt}')
        exp_conv = n
        exp_dt = dt
        break

    u_old = u.copy()
u_explicit = u.copy()

# implicit solution
# initialization
u = np.zeros(ny)
u_old = u.copy()
dt = 1
for n in range(nit):
    apply_boundary(u)

    non_dimensional = (1/Re)*(T*U/H)
    u[1:-1] = ((non_dimensional/dy)*(u[2:]+u[0:-2]) + (dy/dt)*u_old[1:-1])/((2*non_dimensional/dy) + (dy/dt)) # Patankar

    if convergence_criteria(u, u_old) < 1e-6:
        print(f'Implicit converged after {n} iterations, dt = {dt}')
        imp_conv = n
        imp_dt = dt
        break

    u_old = u.copy()
u_implicit = u.copy()

# plot results
fig, ax = plt.subplots(1, 2)
ax[0].plot(np.linspace(0, H, ny), u_explicit, label='Explicit Solution')
ax[0].set_xlabel('Velocity (u)')
ax[0].set_ylabel('y')
ax[0].set_title(f'Explicit Solution - Converged after {exp_conv} iterations, dt = {exp_dt}')
ax[1].plot(np.linspace(0, H, ny), u_implicit, label='Implicit Solution')
ax[1].set_xlabel('Velocity (u)')
ax[1].set_ylabel('y')
ax[1].set_title(f'Implicit Solution - Converged after {imp_conv} iterations, dt = {imp_dt}')
plt.show()