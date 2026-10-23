# https://colab.research.google.com/drive/1pvW9zIZ5qiWjerkWq-iXXTOLWIPkqFLh?usp=sharing
# this could be calculated in 1D using a periodic BC

import numpy as np
from matplotlib import pyplot as plt

rho = 997
mu = 0.001
Cp = 4180
k = 0.6
alpha = k/(rho*Cp)
r1 = 0.03
r2 = 0.12
b = 0.010 # passage depth [m]
nr = 50
dr = (r2-r1)/(nr-1)
ntheta = 100
dtheta = 2*np.pi/(ntheta-1)
g = 9.81
# We have a rectangle => nr x ntheta
rpm = 2800
swirl = 0.82 # swirl fraction (water speed/wheel speed)
omega = 2*np.pi*rpm/60
nit = 20000
h_wall = 120
T_in = 300
T_wall = 330
loss_fraction = 0.04
# Re = 500

r = np.linspace(r1, r2, nr)
theta = np.linspace(0, 2*np.pi, ntheta)
R, THETA = np.meshgrid(r, theta, indexing='ij')

T = np.zeros((nr, ntheta))
# u is velocity in theta, v is velocity in r
# u = R*omega # this is not correct because eater does not flow with the wheel, it is velocity of a point on solid spinning disc
# we have predefined models for this
u = swirl*omega*R
w = u - omega*R # relative to the blade and not outside observer
# w needs to be used in steady state
# dT/dt + v dT/dr + (u/r) dT/dtheta = ... # unsteady eqn has absolute velocity
# dT/dt (lab) = dT/dt (wheel) - Omega * dT/dtheta
# dT/dt (wheel) + u_r dT/dr + ((u - Omega*r)/r) dT/dtheta = ...
v1 = 0.025 # inlet radial speed
v = v1*r1/R # continuity: v*r = constant
# pressure is dervied from prescribed velocity
# dp/dr = rho*u^2/r - 12*mu*v/b^2
p = np.zeros((nr, ntheta))
p_old = p.copy()
for n in range(500):
    p[1:,:] = p_old[0:-1,:] + dr*(rho*u[0:-1,:]**2/R[0:-1,:] - 12*mu*v[0:-1,:]/b**2)
    p_old = p.copy()

def apply_boundary(T):
    T[0,:] = T_in
    T[-1,:] = T[-2,:]
    T[:,-1] = T[:,0] # periodic: last column = first column

T_old = T.copy()
St = loss_fraction*(omega*R)**2*v/(b*Cp*rho)
beta = 2*h_wall*alpha/(b*k)
for n in range(nit):
    # steady state, FDM, Jacobi iteration
    apply_boundary(T)
    T_old = T.copy()

    # interior columns
    TW = T_old[:-2, 1:-1]
    TE = T_old[2:, 1:-1]
    TS = T_old[1:-1, :-2]
    TN = T_old[1:-1, 2:]
    Rc = R[1:-1, 1:-1]
    vc = v[1:-1, 1:-1]
    wc = w[1:-1, 1:-1]
    T[1:-1, 1:-1] = ( vc*TW/dr - (wc/Rc)*TN/dtheta + alpha*((TE + TW)/dr**2 + (TE - TW)/(2*Rc*dr) + (TN + TS)/(Rc**2*dtheta**2)) + beta*T_wall + St[1:-1, 1:-1])/(vc/dr - wc/(Rc*dtheta) + 2*alpha/dr**2 + 2*alpha/(Rc**2*dtheta**2) + beta )

    # seam column (theta = 0 = 2*pi): its neighbours are column -2 and column 1
    TW = T_old[:-2, 0]
    TE = T_old[2:, 0]
    TS = T_old[1:-1, -2]
    TN = T_old[1:-1, 1]
    Rc = R[1:-1, 0]
    vc = v[1:-1, 0]
    wc = w[1:-1, 0]
    T[1:-1, 0] = ( vc*TW/dr - (wc/Rc)*TN/dtheta + alpha*((TE + TW)/dr**2 + (TE - TW)/(2*Rc*dr) + (TN + TS)/(Rc**2*dtheta**2)) + beta*T_wall + St[1:-1, 0])/(vc/dr - wc/(Rc*dtheta) + 2*alpha/dr**2 + 2*alpha/(Rc**2*dtheta**2) + beta )

    residual = np.max(np.abs(T - T_old))

    if np.max(np.abs(T - T_old)) < 1e-7:
      print(f"Converged at n={n}")
      break

# x = r cosθ, y = r sinθ
X = R*np.cos(THETA)
Y = R*np.sin(THETA)
ux = v*np.cos(THETA) - w*np.sin(THETA)
uy = v*np.sin(THETA) + w*np.cos(THETA)
speed = np.sqrt(ux**2 + uy**2)

# pump head
pump_head = p[-1,0]/(rho*g)
print(f"Pump head = {pump_head}m")
# Outlet temperature
T_outlet = T[-1,0]
print(f"Outlet temperature = {T_outlet}")
# Pressure error
p_exact = 0.5*rho*swirl**2*omega**2*(r**2 - r1**2) - 12*mu*v1*r1*np.log(r/r1)/b**2
p_error = (p_exact[-1]-p[-1,0])/p_exact[-1]
print(f"Pressure error = {p_error*100}%")
print(f"Residual = {residual}")

# plot 1: temperature
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, T, shading="gouraud", cmap="inferno")
plt.gca().set_aspect("equal")
plt.colorbar(label="Temperature [K]")
plt.title("Temperature [K]")
plt.xlabel("x [m]")
plt.ylabel("y [m]")

# plot 2: radial pressure rise
plt.figure(figsize=(6, 4.5))
plt.plot(r, p[:,0], "o-", ms=3, label="FDM (forward difference)")
plt.plot(r, p_exact, "--", label="closed form")
plt.xlabel("Radius [m]")
plt.ylabel("Gauge pressure [Pa]")
plt.title("Radial pressure rise")
plt.grid(True, alpha=0.3)
plt.legend()

# plot 3: relative speed and vectors
skip_r, skip_t = nr//10, ntheta//10
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, speed, shading="gouraud", cmap="viridis")
plt.colorbar(label="Relative speed [m/s]")
plt.quiver(X[::skip_r, ::skip_t], Y[::skip_r, ::skip_t],
           ux[::skip_r, ::skip_t], uy[::skip_r, ::skip_t], color="white", scale=60)
plt.gca().set_aspect("equal")
plt.title("Relative speed and vectors")
plt.xlabel("x [m]")
plt.ylabel("y [m]")

plt.show()