import numpy as np
import pygame

# ---------------- Simulation parameters ----------------
N      = 96          # grid resolution (N x N cells)
SCALE  = 6           # pixels per cell
DT     = 0.1         # time step
DIFF   = 0.00005     # dye diffusion rate
VISC   = 0.00005     # kinematic viscosity
ITERS  = 20          # Jacobi solver iterations

SIZE = N + 2         # grid + boundary cells

# Fields: velocity (u, v) and dye density
u  = np.zeros((SIZE, SIZE)); v  = np.zeros((SIZE, SIZE))
u0 = np.zeros((SIZE, SIZE)); v0 = np.zeros((SIZE, SIZE))
dens  = np.zeros((SIZE, SIZE))
dens0 = np.zeros((SIZE, SIZE))

# ---------------- Core solver (Stam's Stable Fluids) ----------------
def set_bnd(b, x):
    """Boundary conditions: walls reflect normal velocity."""
    x[0, :]  = -x[1, :]  if b == 1 else x[1, :]
    x[-1, :] = -x[-2, :] if b == 1 else x[-2, :]
    x[:, 0]  = -x[:, 1]  if b == 2 else x[:, 1]
    x[:, -1] = -x[:, -2] if b == 2 else x[:, -2]
    x[0, 0]   = 0.5 * (x[1, 0] + x[0, 1])
    x[0, -1]  = 0.5 * (x[1, -1] + x[0, -2])
    x[-1, 0]  = 0.5 * (x[-2, 0] + x[-1, 1])
    x[-1, -1] = 0.5 * (x[-2, -1] + x[-1, -2])

def lin_solve(b, x, x0, a, c):
    """Jacobi iteration for the implicit diffusion / pressure equations."""
    for _ in range(ITERS):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1]
                        + x[1:-1, 2:] + x[1:-1, :-2])) / c
        set_bnd(b, x)

def diffuse(b, x, x0, rate):
    """Viscosity / diffusion term: nu * Laplacian(x)."""
    a = DT * rate * N * N
    lin_solve(b, x, x0, a, 1 + 4 * a)

def advect(b, d, d0, u, v):
    """Semi-Lagrangian advection: trace each cell backwards along velocity."""
    I, J = np.meshgrid(np.arange(1, N + 1), np.arange(1, N + 1), indexing="ij")
    x = np.clip(I - DT * N * u[1:-1, 1:-1], 0.5, N + 0.5)
    y = np.clip(J - DT * N * v[1:-1, 1:-1], 0.5, N + 0.5)
    i0 = x.astype(int); j0 = y.astype(int)
    s1 = x - i0; s0 = 1 - s1
    t1 = y - j0; t0 = 1 - t1
    d[1:-1, 1:-1] = (s0 * (t0 * d0[i0, j0]     + t1 * d0[i0, j0 + 1]) +
                     s1 * (t0 * d0[i0 + 1, j0] + t1 * d0[i0 + 1, j0 + 1]))
    set_bnd(b, d)

def project(u, v, p, div):
    """Pressure projection: makes the velocity field divergence-free."""
    div[1:-1, 1:-1] = -0.5 * (u[2:, 1:-1] - u[:-2, 1:-1]
                            + v[1:-1, 2:] - v[1:-1, :-2]) / N
    p.fill(0)
    set_bnd(0, div); set_bnd(0, p)
    lin_solve(0, p, div, 1, 4)
    u[1:-1, 1:-1] -= 0.5 * N * (p[2:, 1:-1] - p[:-2, 1:-1])
    v[1:-1, 1:-1] -= 0.5 * N * (p[1:-1, 2:] - p[1:-1, :-2])
    set_bnd(1, u); set_bnd(2, v)

def step():
    global u, v, u0, v0, dens, dens0
    # velocity step
    diffuse(1, u0, u, VISC); diffuse(2, v0, v, VISC)
    project(u0, v0, u, v)
    advect(1, u, u0, u0, v0); advect(2, v, v0, u0, v0)
    project(u, v, u0, v0)
    # dye step
    diffuse(0, dens0, dens, DIFF)
    advect(0, dens, dens0, u, v)
    dens *= 0.995  # slow fade so the screen doesn't saturate

# ---------------- Interactions ----------------
def add_source(px, py, dx, dy, amount=120.0):
    """Inject dye + momentum where the mouse drags."""
    i = np.clip(px // SCALE, 2, N - 1)
    j = np.clip(py // SCALE, 2, N - 1)
    dens[i-1:i+2, j-1:j+2] += amount * DT
    u[i-1:i+2, j-1:j+2] += dx * 2.0
    v[i-1:i+2, j-1:j+2] += dy * 2.0

def add_vortex(px, py, strength=6.0, radius=14):
    """Spawn a swirling vortex: tangential velocity around the click point."""
    ci, cj = px // SCALE, py // SCALE
    I, J = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing="ij")
    dx = I - ci; dy = J - cj
    r2 = dx * dx + dy * dy + 1e-6
    fall = np.exp(-r2 / (radius ** 2))          # Gaussian falloff
    u[:] += strength * (-dy / np.sqrt(r2)) * fall
    v[:] += strength * ( dx / np.sqrt(r2)) * fall
    dens[:] += 40.0 * fall * DT

# ---------------- Rendering ----------------
def colormap(d):
    """Map dye density to a fire-like color (black -> blue -> cyan -> white)."""
    d = np.clip(d, 0, 1)
    r = np.clip(3 * d - 1.5, 0, 1)
    g = np.clip(3 * d - 0.7, 0, 1)
    b = np.clip(3 * d, 0, 1)
    return (np.dstack([r, g, b]) * 255).astype(np.uint8)

def draw(screen, show_vel):
    img = colormap(dens[1:-1, 1:-1])
    surf = pygame.surfarray.make_surface(img)
    screen.blit(pygame.transform.scale(surf, (N * SCALE, N * SCALE)), (0, 0))
    if show_vel:  # draw velocity arrows every 6 cells
        for i in range(1, N, 6):
            for j in range(1, N, 6):
                x, y = i * SCALE, j * SCALE
                pygame.draw.line(screen, (90, 90, 90), (x, y),
                                 (x + u[i, j] * 40, y + v[i, j] * 40))

# ---------------- Main loop ----------------
def main():
    global dens, u, v
    pygame.init()
    screen = pygame.display.set_mode((N * SCALE, N * SCALE))
    pygame.display.set_caption("Navier-Stokes Fluid | drag: dye  right-click: vortex  V: field  C: clear")
    clock = pygame.time.Clock()
    prev = None; show_vel = False; paused = False

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_c: dens.fill(0); u.fill(0); v.fill(0)
                if e.key == pygame.K_v: show_vel = not show_vel
                if e.key == pygame.K_SPACE: paused = not paused
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                add_vortex(*e.pos)

        if pygame.mouse.get_pressed()[0]:
            pos = pygame.mouse.get_pos()
            if prev:
                add_source(pos[0], pos[1], (pos[0]-prev[0])*0.5, (pos[1]-prev[1])*0.5)
            prev = pos
        else:
            prev = None

        if not paused:
            step()
        draw(screen, show_vel)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()
