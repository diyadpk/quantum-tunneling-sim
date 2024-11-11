import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Gaussian_Wave:
    def __init__(self, N_grid, L, a, V0, w, x0, k0, sigma, t):
        self.t = t  # Time steps for the simulation
        self.L = L  # Length of the grid
        self.N_grid = N_grid  # Number of grid points

        # Create the spatial grid
        self.x = np.linspace(0, self.L, self.N_grid + 1)  # x-grid points
        self.dx = self.x[1] - self.x[0]  # Grid spacing (step size)

        # Initialize the Gaussian wave function (larger sigma -> wider wave)
        self.Psi0 = np.exp(-1 / 2 * (self.x[1:-1] - x0) ** 2 / sigma ** 2) * np.exp(1j * k0 * self.x[1:-1])

        # Normalize the initial state
        norm = np.sum(np.abs(self.Psi0) ** 2 * self.dx)  # Numerically integrate to normalize
        self.Psi0 = self.Psi0 / np.sqrt(norm)

        # Kinetic energy matrix (finite difference for second derivative)
        self.T = -1 / 2 * 1 / self.dx ** 2 * (
            np.diag(-2 * np.ones(self.N_grid - 1)) + np.diag(np.ones(self.N_grid - 2), 1) + np.diag(np.ones(self.N_grid - 2), -1)
        )

        # Define the potential: potential barrier of height V0, starting at 'a', width 'w'
        self.V_flat = np.array([V0 if a < pos < a + w else 0 for pos in self.x[1:-1]])

        # Potential energy matrix (diagonal matrix)
        self.V = np.diag(self.V_flat)

        # Hamiltonian: total energy operator (kinetic + potential)
        self.H = self.T + self.V

        # Solve the eigenvalue problem to get energy eigenvalues and eigenvectors
        self.E, self.psi = np.linalg.eigh(self.H)
        self.psi = self.psi.T  # Eigenvectors as rows

        # Normalize eigenfunctions (important for consistent physics)
        norm = np.sum(np.abs(self.psi) ** 2 * self.dx, axis=1)
        self.psi = self.psi / np.sqrt(norm)

        # Expansion coefficients for the initial state
        self.c_n = np.array([np.sum(np.conj(self.psi[j]) * self.Psi0 * self.dx) for j in range(self.N_grid - 1)])

    # Function to compute the time-dependent wavefunction and animate
    def animation(self):
        # Function for time evolution: returns the wavefunction at time t
        def Psi(t):
            return np.dot(self.psi.T, self.c_n * np.exp(-1j * self.E * t))

        # Set up the figure for animation
        fig, ax = plt.subplots(figsize=(12, 6))  # Larger figure for better visualization
        ax.set_xlim(0, self.L)  # X-axis limit
        ax.set_ylim(-0.5, 0.5)  # Y-axis limit (wider range for better visibility)
        ax.set_title('Quantum Tunneling of Gaussian Wave Packet', fontsize=20)
        ax.set_xlabel('Position ($x$)', fontsize=15)
        ax.set_ylabel('Wavefunction', fontsize=15)

        # Line objects for real and imaginary parts of the wavefunction
        line1, = ax.plot(self.x[1:-1], np.zeros(self.N_grid - 1), lw=2, color="red", label='$\\Re(\\psi)$')
        line2, = ax.plot(self.x[1:-1], np.zeros(self.N_grid - 1), lw=2, color="blue", label='$\\Im(\\psi)$')
        ax.plot(self.x[1:-1], self.V_flat, label='$V(x)$', color='gray')
        ax.legend(fontsize=15)

        # Animation function: updates the wavefunction at each time step
        def animate(t):
            y1 = np.real(Psi(t))  # Real part of wavefunction
            y2 = np.imag(Psi(t))  # Imaginary part of wavefunction
            line1.set_data(self.x[1:-1], y1)  # Update real part plot
            line2.set_data(self.x[1:-1], y2)  # Update imaginary part plot
            return line1, line2

        # Initialize the plot (clearing previous data)
        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            return line1, line2

        # Create the animation using FuncAnimation
        ani = FuncAnimation(fig, animate, frames=self.t, init_func=init,
                            interval=50, blit=True)  # Use blitting for faster animation

        # Show the animation
        plt.show()

        return ani


# Parameters for the simulation with a larger wave (increased sigma)
sigma = 50  # Larger sigma for a bigger wave packet (wider wave)
V0 = 0.3  # Barrier height lower than the wave packet energy to allow tunneling
a = 200  # Start position of the barrier
w = 100  # Narrower barrier to increase the chance of tunneling
k0 = 0.5  # Lower k0 to further reduce the wave packet energy (important for tunneling)

# Create an instance of Gaussian_Wave
wavepacket = Gaussian_Wave(N_grid=500, L=750, a=a, V0=V0, w=w, x0=100, k0=k0, sigma=sigma, t=np.linspace(0., 500, 200))

# Run the animation
wavepacket.animation()
