import numpy as np
import matplotlib.pyplot as plt

N = 200
L = 8.0
R = 2*L
current = 1.0;
height = 0.0;
radius = 1.0;

hz = 2*L / (N-1)
hr = R / (N-1)

def solve_integral(M = 1000):
    z = np.linspace(-L, L, N).reshape(N, 1, 1)
    r = np.linspace(0, R, N).reshape(1, N, 1)
    p = np.linspace(0, 2*np.pi, M).reshape(1, 1, M)

    integrand = np.cos(p) / np.sqrt(r**2 + z**2 + radius**2 - 2*r*radius*np.cos(p))
    integral = np.trapz(integrand, x=p, axis=2)
    phi_a = current * radius / (4.0 * np.pi) * integral
    return phi_a


def draw_plot(grid, title=None, vmin=None, vmax=None):
    # Non-extended.
    #plt.imshow(grid, origin='lower', cmap='jet', extent=[0, R, -L, L], vmin=vmin, vmax=vmax)

    # Extended.
    flipped = np.flip(grid, axis=1)
    glued = np.concatenate((flipped, grid), axis=1)
    plt.imshow(glued, origin='lower', cmap='jet', extent=[-R, R, -L, L], vmin=vmin, vmax=vmax)

    plt.title(title)
    plt.xlabel("r")
    plt.ylabel("z")
    plt.colorbar()
    plt.show()
    plt.clf()


def plotter(x, numerical, analytical=None, title=None):
    plt.plot(x, numerical, label='Numerical', color='red')
    if analytical is not None:
        plt.plot(x, analytical, label='Analytical', color='black')

    plt.title(title)
    plt.legend()
    plt.show()


import data
a_phi = solve_integral(100)
draw_plot(a_phi, title="Integral Solution")
draw_plot(data.grid, title="Numerical Solution")

error_grid = np.abs((a_phi - data.grid) / a_phi) * 100.0
draw_plot(error_grid, title="abs[(Analytical - Numerical) / Analytical] * 100", vmin=0, vmax=100)


x = np.linspace(0, R, N)
plotter(x, data.grid[0,:], analytical=a_phi[0,:], title="A_theta at r-axis (z=0)")

nn = round((L+1/hz))
plotter(x, data.grid[nn,:], analytical=a_phi[nn,:], title="A_theta at r-axis (z=1)")


