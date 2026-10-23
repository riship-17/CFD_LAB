import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
Lx = 30 # Length of domain
th = 0.1 # thickness of wall
Dy_hot = 5 # Diameter of each cylinder
Dy_cold = 5 # Diameter of each cylinder
Ly = Dy_hot + Dy_cold + th # Length of domain
nx = 200
ny_hot = 20
ny_wall = 3 # solid interface
ny_cold = 30
ny = ny_hot + ny_wall + ny_cold
x_array = np.linspace(0, Lx, nx)
# Each zone gets its own y-coordinates spanning ITS OWN physical thickness
# (Dy_cold, th, Dy_hot), not the full domain height Ly. Concatenating them
# keeps the plotted grid aligned with the true cold/wall/hot geometry.
y_cold_coords = np.linspace(0, Dy_cold, ny_cold)
y_wall_coords = np.linspace(Dy_cold, Dy_cold + th, ny_wall)
y_hot_coords  = np.linspace(Dy_cold + th, Ly, ny_hot)
y_array = np.concatenate([y_cold_coords, y_wall_coords, y_hot_coords])
X, Y = np.meshgrid(x_array, y_array)
dx = Lx/(nx-1)
# dy in each zone = that zone's own thickness / (its node count - 1),
# NOT the full domain height Ly. (Ly/(ny_wall-1) made dy_wall ~100x too
# large, which made the wall's D/dy^2 conduction term ~10,000x too weak
# and effectively decoupled the hot and cold channels.)
dy_hot = Dy_hot/(ny_hot-1)
dy_wall = th/(ny_wall-1)
dy_cold = Dy_cold/(ny_cold-1)

# Flow and Heat properties
# Water fluid and copper solid
k_f = 0.6 # thermal conductivity of water [W/mK]
k_s = 237 # thermal conductivity of aluminium [W/mK]
Pr = 7.0 # Approximate Prandtl number for water
Re_hot = 50 # Reynolds number of hot fluid
Re_cold = 50 # Reynolds number of cold fluid
T_hot_in = 5.0 # non-dimensional hot fluid inlet temperature
T_cold_in = -5.0 # non-dimensional cold fluid inlet temperature
dp_cold = 1.0 # pressure drop in cold channel
dp_hot = 1.0 # pressure drop in hot channel
mu_cold = 1.0 # dynamic viscosity of cold fluid
mu_hot = 1.0 # dynamic viscosity of hot fluid
counter_flow = True # counter flow if True, parallel flow if False

# Simulation parameters
ni = 2000000 # number of iterations. With this explicit (CFL-limited) scheme,
             # dt ends up small enough that the flow needs roughly a million+
             # steps just to advect once through the domain and let the
             # boundary layer develop - 10,000 wasn't remotely enough. This
             # will take real wall-clock time (tens of minutes); the tol
             # check below will still break out early if it converges sooner.
tol = 0.5e-5 # convergence tolerance

def velocity_field(ny_cold, ny_wall, ny_hot, nx, counter_flow):
    """Parabolic (Poiseuille) profile in each channel, exactly zero in the wall rows."""
    ny = ny_cold + ny_wall + ny_hot
    u = np.zeros((ny, nx))

    # 1. Physical coordinates measured from the centerline of each channel (-R to +R)
    R_hot = Dy_hot / 2.0
    R_cold = Dy_cold / 2.0
    y_cold = np.linspace(-R_cold, R_cold, ny_cold)
    y_hot  = np.linspace(-R_hot, R_hot, ny_hot)

    # 2. Literal Planar Poiseuille Equation: u(y) = (ΔP / 2μL) * (R² - y²)
    # (Denominator is 2.0 for flat planar channels, use 4.0 for cylindrical pipes)
    profile_cold = (dp_cold / (2.0 * mu_cold * Lx)) * (R_cold**2 - y_cold**2)
    profile_hot  = (dp_hot  / (2.0 * mu_hot * Lx))  * (R_hot**2  - y_hot**2)

    # 3. Flow direction logic for the cold stream
    sign_cold = -1.0 if counter_flow else 1.0

    # 4. Map profiles into the global velocity array (wall rows remain exactly zero)
    u[:ny_cold] = sign_cold * profile_cold[:, None]
    u[ny_cold + ny_wall:] = profile_hot[:, None]

    return u

u = velocity_field(ny_cold, ny_wall, ny_hot, nx, counter_flow)
v = np.zeros_like(u)  # No vertical velocity component

# # Plot the velocity field with a background color mesh and quiver vectors
# plt.figure(figsize=(12, 5))
# # 1. Background color mesh reveals the continuous parabolic gradients instantly
# mesh = plt.pcolormesh(X, Y, u, cmap='RdBu_r', shading='auto', alpha=0.8)
# cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.15)
# cbar.set_label('Velocity magnitude (u)')
# # 2. Sub-sample the grid (skipping points) to prevent vector overcrowding
# # Takes every 5th vertical point and every 25th horizontal point
# skip = (slice(None, None, 5), slice(None, None, 25))
# # 3. Plot vectors using auto-scaling ('width') instead of forcing spatial 'xy' coordinates
# plt.quiver(X[skip], Y[skip], u[skip], v[skip], 
#            color='black', units='width', scale=15, 
#            width=0.0015, headwidth=4)
# # 4. Draw a visual boundary for your solid copper interface wall
# plt.axhspan(Dy_cold, Dy_cold + th, color='gray', alpha=0.6, label='Solid Interface')
# plt.title("Poiseuille Velocity Field: Parabolic Gradients & Vectors")
# plt.xlabel("Channel Length (x)")
# plt.ylabel("Channel Height (y)")
# plt.xlim(0, Lx)
# plt.ylim(0, Ly)
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.show()

# Temperature field initialization
T = np.zeros((ny, nx))
T[:ny_cold, :] = T_cold_in
T[ny_cold + ny_wall:, :] = T_hot_in

# Diffusivity field D(x,y) is piecewise constant in each region (cold, wall, hot)
D_cold_val = 1.0 / (Re_cold * Pr)
D_hot_val = 1.0 / (Re_hot * Pr)
D_wall_val = (k_s / k_f) * ((D_hot_val + D_cold_val) / 2.0)
D = np.zeros((ny, nx))
D[:ny_cold, :] = D_cold_val
D[ny_cold:ny_cold + ny_wall, :] = D_wall_val
D[ny_cold + ny_wall:, :] = D_hot_val

def apply_bc():
    # Temperature Dirichlet boundary conditions at the inlets of the channels
    # Temperature Neumann boundary conditions at the outlets of the channels (zero gradient)
    # With counterflow, cold one switches direction, hot one always goes from left to right
    if counter_flow:
        # Counter-flow: Hot enters from left (x=0), Cold enters from right (x=Lx)
        T[:ny_cold, -1] = T_cold_in       # inlet of cold channel
        T[:ny_cold, 0] = T[:ny_cold, 1]  # outlet of cold channel
    else:
        # Co-flow (Parallel): Both fluids enter from the left edge (x=0)
        T[:ny_cold, 0] = T_cold_in       # Left edge of cold channel
        T[:ny_cold, -1] = T[:ny_cold, -2]       # outlet of cold channel
    T[ny_cold + ny_wall:, 0] = T_hot_in   # Left edge of hot channel
    T[ny_cold + ny_wall:, -1] = T[ny_cold + ny_wall:, -2]  # outlet of hot channel
    # Wall Neumann boundary conditions (zero gradient) at the solid interface
    # interface BC can be termed as continuity BC
    T[ny_cold:ny_cold + ny_wall, 0] = T[ny_cold:ny_cold + ny_wall, 1] # Left edge of wall
    T[ny_cold:ny_cold + ny_wall, -1] = T[ny_cold:ny_cold + ny_wall, -2] # Right edge of wall
    T[0, :] = T[1, :]  # Bottom edge of cold channel (whole domain insulated)
    T[-1, :] = T[-2, :]  # Top edge of hot channel (whole domain insulated)

# Create a 2D column vector of dy^2 that correctly matches the interior grid regions (ny-2 rows)
# Because dy value is different in different regions, we need to create a 2D array that has the correct dy values for each interior row
dy_array = np.zeros((ny - 2, 1))
dy_array[:ny_cold - 1, 0] = dy_cold
dy_array[ny_cold - 1:ny_cold + ny_wall - 1, 0] = dy_wall
dy_array[ny_cold + ny_wall - 1:, 0] = dy_hot

def fdm_step():
    Tc = T[1:-1, 1:-1]  # Interior points
    Te = T[1:-1, 2:]    # East neighbors
    Tw = T[1:-1, :-2]   # West neighbors
    Tn = T[2:, 1:-1]    # North neighbors
    Ts = T[:-2, 1:-1]   # South neighbors

    # Non-dimensionalized diffusivity D(x,y) for FDM
    # Dc = D[1:-1, 1:-1]
    # For FVM, we would have taken the harmonic mean of Dc and at half nodes
    D_e = D[1:-1, 2:]
    D_w = D[1:-1, :-2]
    D_n = D[2:, 1:-1]
    D_s = D[:-2, 1:-1]

    diffusion_x = (D_e * (Te - Tc) - D_w * (Tc - Tw)) / dx**2
    diffusion_y = (D_n * (Tn - Tc) - D_s * (Tc - Ts)) / dy_array**2

    uc = u[1:-1, 1:-1]

    dTdx_upwind = np.where(uc >= 0, (Tc - Tw) / dx, (Te - Tc) / dx)
    convection = uc * dTdx_upwind

    # Pseudo-time stepping for the transient solution of the steady-state problem
    # Determine maximum stable time step via Courant-Friedrichs-Lewy (CFL) limit
    max_D = np.max(D)
    max_u = np.max(np.abs(uc))
    min_dy = min(dy_cold, dy_wall, dy_hot)
    dt = 0.4 / (max_D * (2/dx**2 + 2/min_dy**2) + max_u/dx)

    T_new = T.copy()
    T_new[1:-1, 1:-1] = Tc + dt * (diffusion_x + diffusion_y - convection)

    return T_new

######################
# Main simulation loop
apply_bc()

for it in range(ni):
    T_old = T.copy()
    T = fdm_step()

    apply_bc()  # Re-apply boundary conditions after updating interior points

    # Compute maximum absolute change for convergence
    error = np.max(np.abs(T - T_old))

    if it % 500 == 0 or it == ni - 1:
        print(f"Iteration {it}, Error: {error:.6e}")

    if error < tol:
        print(f"Converged successfully at iteration {it} with error {error:.6e}")
        break

# Plotting the Final Temperature Field
plt.figure(figsize=(12, 5))
mesh = plt.pcolormesh(X, Y, T, cmap='RdBu', shading='auto')
cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.15)
cbar.set_label('Non-dimensional Temperature (T)')
plt.axhspan(Dy_cold, Dy_cold + th, color='gray', alpha=0.5, label='Solid Wall Interface')
plt.title("Steady-State Conjugate Heat Transfer: Temperature Distribution")
plt.xlabel("Channel Length (x)")
plt.ylabel("Channel Height (y)")
plt.xlim(0, Lx)
plt.ylim(0, Ly)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()