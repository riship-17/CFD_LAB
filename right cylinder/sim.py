"""
Immersed boundary method is used to treat fluid-solid interface at cylinder surface

https://github.com/sustielz/IB-python/blob/main/2D_benchmark.ipynb

A smooth mask alpha(x,y) is 1 inside the solid and 0 outside. A penalty
term -alpha*u/eta in the momentum equation forces u -> 0 inside the
airfoil. Forces are estimated from that penalty term and are
QUALITATIVE only (teaching level).
"""

import numpy as np
from matplotlib import pyplot as plt

D = 1                  # length scale = cylinder diameter
Lx = 25
Ly = 5
nx = 500
ny = 100
dx = Lx/nx             # 0.05 D
dy = Ly/ny
Cx = Lx/3
Cy = Ly/2
Cr = 0.5*D             # radius = D/2
nit = 50000
Co = 0.1
Re = 150
u_inlet = 1            # U, nondimensional
dt = Co * min(dx, dy) / u_inlet     # 0.0125 (in units of D/U)
eta = max(0.5 * dt, 1e-5)           # nondimensional penalty parameter
alpha_p = 0.3

def apply_boundary(u, v, p):
    # left
    u[:,0] = 1
    v[:,0]=0
    p[:,0] = p[:,1]
    # right
    u[:,-1] = u[:,-2]
    v[:,-1] = v[:,-2]
    p[:,-1] = 0
    # top symmetry
    u[0,:] = u[1,:]
    v[0,:] = 0
    p[0,:] = p[1,:]
    # bottom symmetry
    u[-1,:] = u[-2,:]
    v[-1,:] = 0
    p[-1,:] = p[-2,:]

alpha = np.zeros((ny, nx))
# draw cylinder
# alpha is 1 inside cylinder, 0 outside it in fluid region
x = np.linspace(0, (nx - 1) * dx, nx)
y = np.linspace(0, (ny - 1) * dy, ny)
X, Y = np.meshgrid(x, y)
mask = (X - Cx)**2 + (Y - Cy)**2 < Cr**2
alpha[mask] = 1
# mask plot
# plt.figure(figsize=(10, 2))
# plt.imshow(alpha, extent=[0, Lx, 0, Ly], origin='lower', cmap='gray')
# plt.title('Cylinder immersed in fluid')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# initial conditions
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
pc = np.zeros((ny, nx))
u_old = u.copy()
v_old = v.copy()
p_old = p.copy()

Cl = np.zeros(nit)

for n in range(nit):
    apply_boundary(u, v, p)
    u_old = u.copy()
    v_old = v.copy()
    p_old = p.copy()

    # 1. u*: upwind convection + (1/Re) diffusion - grad p, penalty implicit
    u[1:-1,1:-1] = (u_old[1:-1,1:-1] + dt*(
        - np.maximum(u_old[1:-1,1:-1],0)*(u_old[1:-1,1:-1] - u_old[1:-1,:-2])/dx
        + np.minimum(u_old[1:-1,1:-1],0)*(u_old[1:-1,1:-1] - u_old[1:-1,2:])/dx
        - np.maximum(v_old[1:-1,1:-1],0)*(u_old[1:-1,1:-1] - u_old[:-2,1:-1])/dy
        + np.minimum(v_old[1:-1,1:-1],0)*(u_old[1:-1,1:-1] - u_old[2:,1:-1])/dy
        + (1/Re)*((u_old[1:-1,2:] - 2*u_old[1:-1,1:-1] + u_old[1:-1,:-2])/dx**2
                + (u_old[2:,1:-1] - 2*u_old[1:-1,1:-1] + u_old[:-2,1:-1])/dy**2)
        - (p[1:-1,2:] - p[1:-1,:-2])/(2*dx)
    )) / (1 + dt*alpha[1:-1,1:-1]/eta)
    # 2. v*
    v[1:-1,1:-1] = (v_old[1:-1,1:-1] + dt*(
        - np.maximum(u_old[1:-1,1:-1],0)*(v_old[1:-1,1:-1] - v_old[1:-1,:-2])/dx
        + np.minimum(u_old[1:-1,1:-1],0)*(v_old[1:-1,1:-1] - v_old[1:-1,2:])/dx
        - np.maximum(v_old[1:-1,1:-1],0)*(v_old[1:-1,1:-1] - v_old[:-2,1:-1])/dy
        + np.minimum(v_old[1:-1,1:-1],0)*(v_old[1:-1,1:-1] - v_old[2:,1:-1])/dy
        + (1/Re)*((v_old[1:-1,2:] - 2*v_old[1:-1,1:-1] + v_old[1:-1,:-2])/dx**2
                + (v_old[2:,1:-1] - 2*v_old[1:-1,1:-1] + v_old[:-2,1:-1])/dy**2)
        - (p[2:,1:-1] - p[:-2,1:-1])/(2*dy)
    )) / (1 + dt*alpha[1:-1,1:-1]/eta)
    # 3. pressure correction: laplacian(p') = div(u*)/dt
    div = np.zeros((ny, nx))
    div[1:-1,1:-1] = (u[1:-1,2:] - u[1:-1,:-2])/(2*dx) + (v[2:,1:-1] - v[:-2,1:-1])/(2*dy)
    pc[:,:] = 0
    for k in range(50):
        pc[:,0] = pc[:,1]; pc[0,:] = pc[1,:]; pc[-1,:] = pc[-2,:]; pc[:,-1] = 0
        pc[1:-1,1:-1] = ((pc[1:-1,2:] + pc[1:-1,:-2])*dy**2 + (pc[2:,1:-1] + pc[:-2,1:-1])*dx**2
                         - dx**2*dy**2*div[1:-1,1:-1]/dt) / (2*(dx**2 + dy**2))
    # 4. correct velocity and pressure
    u[1:-1,1:-1] = u[1:-1,1:-1] - dt*(pc[1:-1,2:] - pc[1:-1,:-2])/(2*dx)/(1 + dt*alpha[1:-1,1:-1]/eta)
    v[1:-1,1:-1] = v[1:-1,1:-1] - dt*(pc[2:,1:-1] - pc[:-2,1:-1])/(2*dy)/(1 + dt*alpha[1:-1,1:-1]/eta)
    p = p_old + alpha_p*pc

    Fy = np.sum(alpha*v)/eta*dx*dy
    Cl[n] = 2*Fy/(u_inlet**2*D) # force on the fluid by the penalty term, sign flipped for the body below

    if n%500 == 0:
        print(f"Still solving n={n}")

plt.figure(figsize=(10, 2))
plt.plot(np.linspace(dt, dt*nit, nit), Cl)
plt.ylabel('Cl')
plt.title('Cylinder immersed in fluid')
plt.xlabel('nt')
plt.show()

# ---------- vorticity ----------
w = np.gradient(v, dx, axis=1) - np.gradient(u, dy, axis=0) # dv/dx - du/dy
w[mask] = np.nan # hide cylinder interior
plt.figure(figsize=(12, 3))
plt.imshow(w, origin='lower', extent=[0, Lx, 0, Ly], cmap='seismic', vmin=-2, vmax=2)
plt.colorbar(label='vorticity')
plt.title(f'Vorticity at t = {nit*dt:.1f}')
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# ---------- Strouhal number from Cl ----------
# St = f*D/U ; time is already scaled by D/U, so St = f of the Cl oscillation
start = nit//2                              # discard start-up transient
cl = Cl[start:] - np.mean(Cl[start:])
print("Cl amplitude (last half):", np.max(np.abs(cl)))

if np.max(np.abs(cl)) > 1e-3:               # only if Cl actually oscillates
    npad = 16*len(cl)                       # zero-padding -> finer frequency grid
    freq = np.fft.rfftfreq(npad, d=dt)
    amp = np.abs(np.fft.rfft(cl, n=npad))
    f = freq[np.argmax(amp[1:]) + 1]        # skip the zero-frequency bin
    St = f*D/u_inlet
    print(f"Shedding frequency f = {f:.4f}, St = {St:.4f}  (literature ~0.183 at Re = 150)")

    plt.figure(figsize=(6, 3))
    plt.plot(freq, amp)
    plt.xlim(0, 1)
    plt.xlabel('frequency (U/D)')
    plt.ylabel('|FFT(Cl)|')
    plt.title('Cl spectrum, peak = St')
    plt.show()
else:
    print("Cl is flat: no shedding developed, so St cannot be computed.")