"""
Immersed boundary method is used to treat fluid-solid interface at airfoil surface
(NACA 00xx, penalty / volume penalization, steady state, Cl vs angle of attack)

Nondimensional NS: du/dt + (u.grad)u = -grad p + (1/Re) lap u - alpha*u/eta
Length scale = chord c, velocity scale = U
"""

import numpy as np
from matplotlib import pyplot as plt

c = 1                  # chord = length scale
Lx = 6                 # 6 c
Ly = 3                 # 3 c
nx = 400
ny = 200               # dx = dy = 0.015 c
dx = Lx/nx
dy = Ly/ny
xle = Lx/4             # leading edge x position
Cy = Ly/2              # centerline y position
tk = 0.12              # NACA 00xx thickness ratio
nit = 15000            # max steps per angle
Co = 0.2
Re = 100
u_inlet = 1            # U, nondimensional
dt = Co * min(dx, dy) / u_inlet
eta = 0.1*dt           # nondimensional penalty parameter
alpha_p = 1.0          # time-marching: take the full pressure correction
tol_cl = 1e-6          # steady state: Cl change over 500 steps below this
aoa_list = [4, 8]      # angles of attack in degrees

def apply_boundary(u, v, p):
    # left (inlet)
    u[:,0] = 1
    v[:,0] = 0
    p[:,0] = p[:,1]
    # right (outlet)
    u[:,-1] = u[:,-2]
    v[:,-1] = v[:,-2]
    p[:,-1] = 0
    # bottom symmetry
    u[0,:] = u[1,:]
    v[0,:] = 0
    p[0,:] = p[1,:]
    # top symmetry
    u[-1,:] = u[-2,:]
    v[-1,:] = 0
    p[-1,:] = p[-2,:]


x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dy, ny)
X, Y = np.meshgrid(x, y)

# exact Poisson solver, built once (the grid never changes)
# interior equation:  Ay @ P + P @ Ax = B
ni, nj = ny-2, nx-2
Ax = (np.diag(-2*np.ones(nj)) + np.diag(np.ones(nj-1), 1) + np.diag(np.ones(nj-1), -1))/dx**2
Ax[0, 0] = -1/dx**2                          # inlet: zero gradient (outlet pc = 0 is the plain -2)
Ay = (np.diag(-2*np.ones(ni)) + np.diag(np.ones(ni-1), 1) + np.diag(np.ones(ni-1), -1))/dy**2
Ay[0, 0] = -1/dy**2; Ay[-1, -1] = -1/dy**2   # top and bottom: zero gradient
ly, Qy = np.linalg.eigh(Ay)
lx, Qx = np.linalg.eigh(Ax)
den = ly[:, None] + lx[None, :]              # all negative, no division by zero

# initial conditions (first angle only; later angles start from the previous solution)
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
pc = np.zeros((ny, nx))
u[:,:] = u_inlet

Cl_aoa = np.zeros(len(aoa_list))
Cd_aoa = np.zeros(len(aoa_list))

for ia, aoa_deg in enumerate(aoa_list):
    aoa = aoa_deg*np.pi/180

    # ---------- airfoil mask for this angle ----------
    # coordinates in the airfoil frame (origin at leading edge, x along the chord)
    # positive aoa = nose up
    xb = (X - xle)*np.cos(aoa) - (Y - Cy)*np.sin(aoa)
    yb = (X - xle)*np.sin(aoa) + (Y - Cy)*np.cos(aoa)
    xc = np.clip(xb/c, 0, 1)
    yt = 5*tk*(0.2969*np.sqrt(xc) - 0.1260*xc - 0.3516*xc**2 + 0.2843*xc**3 - 0.1036*xc**4)*c
    mask = (xb >= 0) & (xb <= c) & (np.abs(yb) < yt)
    alpha = np.zeros((ny, nx))
    alpha[mask] = 1
    u[mask] = 0
    v[mask] = 0

    Cl = np.zeros(nit)
    Cd = np.zeros(nit)

    for n in range(nit):
        apply_boundary(u, v, p)
        u_old = u.copy()
        v_old = v.copy()
        p_old = p.copy()

        # 1. u*: central convection + (1/Re) diffusion - grad p, penalty implicit
        u[1:-1,1:-1] = (u_old[1:-1,1:-1] + dt*(
            - u_old[1:-1,1:-1]*(u_old[1:-1,2:] - u_old[1:-1,:-2])/(2*dx)
            - v_old[1:-1,1:-1]*(u_old[2:,1:-1] - u_old[:-2,1:-1])/(2*dy)
            + (1/Re)*((u_old[1:-1,2:] - 2*u_old[1:-1,1:-1] + u_old[1:-1,:-2])/dx**2
                    + (u_old[2:,1:-1] - 2*u_old[1:-1,1:-1] + u_old[:-2,1:-1])/dy**2)
            - (p[1:-1,2:] - p[1:-1,:-2])/(2*dx)
        )) / (1 + dt*alpha[1:-1,1:-1]/eta)
        # 2. v*
        v[1:-1,1:-1] = (v_old[1:-1,1:-1] + dt*(
            - u_old[1:-1,1:-1]*(v_old[1:-1,2:] - v_old[1:-1,:-2])/(2*dx)
            - v_old[1:-1,1:-1]*(v_old[2:,1:-1] - v_old[:-2,1:-1])/(2*dy)
            + (1/Re)*((v_old[1:-1,2:] - 2*v_old[1:-1,1:-1] + v_old[1:-1,:-2])/dx**2
                    + (v_old[2:,1:-1] - 2*v_old[1:-1,1:-1] + v_old[:-2,1:-1])/dy**2)
            - (p[2:,1:-1] - p[:-2,1:-1])/(2*dy)
        )) / (1 + dt*alpha[1:-1,1:-1]/eta)

        # force on the body from the penalty term (taken from the penalised predictor)
        Fx = np.sum(alpha*u)/eta*dx*dy
        Fy = np.sum(alpha*v)/eta*dx*dy
        Cd[n] = 2*Fx/(u_inlet**2*c)
        Cl[n] = 2*Fy/(u_inlet**2*c)

        # 3. pressure correction: laplacian(p') = div(u*)/dt   (exact solve)
        div = np.zeros((ny, nx))
        div[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,:-2])/(2*dx) + (v[2:,1:-1] - v[:-2,1:-1])/(2*dy)
        pc[1:-1,1:-1] = Qy @ ((Qy.T @ (div[1:-1,1:-1]/dt) @ Qx)/den) @ Qx.T
        pc[:,0] = pc[:,1]; pc[0,:] = pc[1,:]; pc[-1,:] = pc[-2,:]; pc[:,-1] = 0
        # 4. correct velocity and pressure
        u[1:-1,1:-1] = u[1:-1,1:-1] - dt*(pc[1:-1,2:] - pc[1:-1,:-2])/(2*dx)
        v[1:-1,1:-1] = v[1:-1,1:-1] - dt*(pc[2:,1:-1] - pc[:-2,1:-1])/(2*dy)
        p = p_old + alpha_p*pc

        # 5. Continuity residual: divergence of corrected velocity
        div_corr = np.zeros((ny, nx))
        div_corr[1:-1, 1:-1] = (
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx)
            + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)
        )
        # Evaluate only interior fluid cells, excluding the airfoil
        fluid_div = div_corr[1:-1, 1:-1][~mask[1:-1, 1:-1]]
        # RMS and maximum absolute continuity residuals
        cont_rms = np.sqrt(np.mean(fluid_div**2))
        cont_max = np.max(np.abs(fluid_div))

        if n % 500 == 0:
            print(f"aoa = {aoa_deg}, n = {n}, res = {cont_max:.2e}, Cl = {Cl[n]:.4f}, Cd = {Cd[n]:.4f}")

        # steady state: Cl no longer changing
        if n > 3000 and abs(Cl[n] - Cl[n-500]) < tol_cl:
            break
        # if cont_max < 1e-6:
        #     break

    Cl_aoa[ia] = np.mean(Cl[max(n-100, 0):n+1])
    Cd_aoa[ia] = np.mean(Cd[max(n-100, 0):n+1])
    print(f"aoa = {aoa_deg} deg, steps = {n+1}, residual = {cont_max:.2e}, Cl = {Cl_aoa[ia]:.4f}, Cd = {Cd_aoa[ia]:.4f}")

# ---------- Cl vs aoa ----------
aoa_arr = np.array(aoa_list)
plt.figure(figsize=(6, 4))
plt.plot(aoa_arr, Cl_aoa, 'o-')
plt.xlabel('angle of attack (deg)')
plt.ylabel('Cl')
plt.title(f'Cl vs angle of attack, Re = {Re}')
plt.grid(True)
plt.show()

# ---------- contour plots for the last angle ----------
speed = np.sqrt(u**2 + v**2)
speed = np.ma.masked_where(mask, speed)       # hide airfoil interior
pm = np.ma.masked_where(mask, p)

plt.figure(figsize=(12, 3))
plt.contourf(X, Y, speed, levels=30, cmap='jet')
plt.colorbar(label='|velocity| / U')
plt.title(f'Velocity magnitude, aoa = {aoa_list[-1]} deg')
plt.xlabel('x / c')
plt.ylabel('y / c')
plt.gca().set_aspect('equal')
plt.show()

plt.figure(figsize=(12, 3))
plt.contourf(X, Y, pm, levels=30, cmap='seismic')
plt.colorbar(label='p / (rho U^2)')
plt.title(f'Pressure, aoa = {aoa_list[-1]} deg')
plt.xlabel('x / c')
plt.ylabel('y / c')
plt.gca().set_aspect('equal')
plt.show()