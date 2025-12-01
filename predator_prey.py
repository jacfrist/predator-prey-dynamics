"""
Predator-Prey Dynamics Simulation using Lotka-Volterra Equations
Jacqueline Frist
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class LotkaVolterraModel:
    """
    Implements the Lotka-Volterra model.

    alpha: Prey birth rate
    beta: Predation rate (how often predators catch prey)
    gamma: Predator death rate
    delta: Predator reproduction rate (based on eating prey)
    """

    def __init__(self, alpha, beta, gamma, delta):
        self.alpha = alpha 
        self.beta = beta  
        self.gamma = gamma 
        self.delta = delta

    def derivatives(self, x, y):
        """
        Calculate dx/dt and dy/dt at given populations.

        x: Prey population
        y: Predator population

        Returns tuple: (dx/dt, dy/dt)
        """
        dxdt = self.alpha * x - self.beta * x * y
        dydt = self.delta * x * y - self.gamma * y
        return dxdt, dydt

    def euler_method(self, x0, y0, t_max, dt):
        """
        Solve the Lotka-Volterra equations using Euler's method.

        x0: Initial prey population
        y0: Initial predator population
        t_max: Maximum time to simulate
        dt: Time step size

        Returns tuple: (t_array, x_array, y_array)
        """
        n_steps = int(t_max / dt)
        t = np.zeros(n_steps)
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)

        # Initial conditions
        x[0] = x0
        y[0] = y0
        t[0] = 0

        # Euler's method
        for i in range(1, n_steps):
            dxdt, dydt = self.derivatives(x[i-1], y[i-1])
            x[i] = x[i-1] + dt * dxdt
            y[i] = y[i-1] + dt * dydt
            t[i] = t[i-1] + dt

            # Prevent negative results
            x[i] = max(0, x[i])
            y[i] = max(0, y[i])

        return t, x, y

    def equilibrium_points(self):
        """
        Calculates equilibrium points where dx/dt = 0 and dy/dt = 0.

        Returns list of tuples: [(x1, y1), (x2, y2), ...]
        """
        # Extinction equilibrium 
        eq1 = (0, 0)

        # Coexistence equilibrium
        # y = α/β
        # x = γ/δ
        eq2 = (self.gamma / self.delta, self.alpha / self.beta)

        return [eq1, eq2]

    def jacobian(self, x, y):
        """
        Calculate the Jacobian matrix at point (x, y).

        x:  Prey population
        y:  Predator population

        Returns array (2x2 Jacobian matrix)
        """
        J = np.array([
            [self.alpha - self.beta * y, -self.beta * x],
            [self.delta * y, self.delta * x - self.gamma]
        ])
        return J

    def stability_analysis(self):
        """
        Perform stability analysis at equilibrium points.

        Returns list of dicts: Information about each equilibrium point
        """
        equilibria = self.equilibrium_points()
        results = []

        for eq in equilibria:
            x_eq, y_eq = eq
            J = self.jacobian(x_eq, y_eq)
            eigenvalues = np.linalg.eigvals(J)

            # Determine stability
            real_parts = np.real(eigenvalues)
            imag_parts = np.imag(eigenvalues)

            if np.allclose(real_parts, 0):
                stability = "Center (neutral stability - periodic orbits)"
            elif np.all(real_parts < 0):
                stability = "Stable (attracting)"
            elif np.any(real_parts > 0):
                stability = "Unstable (repelling)"
            else:
                stability = "Unknown"

            results.append({
                'equilibrium': eq,
                'jacobian': J,
                'eigenvalues': eigenvalues,
                'stability': stability
            })

        return results


def plot_time_series(t, x, y, title="Population Dynamics Over Time"):
    """
    Plot prey and predator populations over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, x, 'b-', label='Prey', linewidth=2)
    plt.plot(t, y, 'r-', label='Predators', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Population', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_phase_portrait(model, trajectories, show_vector_field=True):
    """
    Plot phase portrait

    model: The model instance
    trajectories: List of (t, x, y) tuples for different initial conditions
    show_vector_field: Whether to show the direction field
    """
    plt.figure(figsize=(10, 8))

    # Plot vector field
    if show_vector_field:
        x_range = np.linspace(0, max([max(x) for _, x, _ in trajectories]) * 1.2, 20)
        y_range = np.linspace(0, max([max(y) for _, _, y in trajectories]) * 1.2, 20)
        X, Y = np.meshgrid(x_range, y_range)

        dX = np.zeros_like(X)
        dY = np.zeros_like(Y)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                dxdt, dydt = model.derivatives(X[i, j], Y[i, j])
                dX[i, j] = dxdt
                dY[i, j] = dydt

        # Normalize arrows
        magnitude = np.sqrt(dX**2 + dY**2)
        magnitude[magnitude == 0] = 1 
        dX_norm = dX / magnitude
        dY_norm = dY / magnitude

        plt.quiver(X, Y, dX_norm, dY_norm, magnitude,
                  alpha=0.4, cmap='viridis', scale=25)

    # Plot trajectories
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    for idx, (t, x, y) in enumerate(trajectories):
        plt.plot(x, y, linewidth=2, color=colors[idx],
                label=f'Initial: ({x[0]:.1f}, {y[0]:.1f})')
        # Starting point
        plt.plot(x[0], y[0], 'o', color=colors[idx], markersize=8)

    # Plot equilibrium points
    equilibria = model.equilibrium_points()
    for eq in equilibria:
        plt.plot(eq[0], eq[1], 'k*', markersize=15,
                label=f'Equilibrium: ({eq[0]:.2f}, {eq[1]:.2f})')

    plt.xlabel('Prey Population (x)', fontsize=12)
    plt.ylabel('Predator Population (y)', fontsize=12)
    plt.title('Phase Portrait: Predator-Prey Dynamics', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def compare_step_sizes(model, x0, y0, t_max, step_sizes):
    """
    Compares the effect of different step sizes.
    """
    fig, axes = plt.subplots(2, len(step_sizes), figsize=(15, 8))

    for idx, dt in enumerate(step_sizes):
        t, x, y = model.euler_method(x0, y0, t_max, dt)

        # Time series plot
        axes[0, idx].plot(t, x, 'b-', label='Prey', linewidth=1.5)
        axes[0, idx].plot(t, y, 'r-', label='Predators', linewidth=1.5)
        axes[0, idx].set_title(f'dt = {dt}', fontweight='bold')
        axes[0, idx].set_xlabel('Time')
        axes[0, idx].set_ylabel('Population')
        axes[0, idx].legend()
        axes[0, idx].grid(True, alpha=0.3)

        # Phase portrait
        axes[1, idx].plot(x, y, 'g-', linewidth=1.5)
        axes[1, idx].plot(x[0], y[0], 'go', markersize=8)
        equilibria = model.equilibrium_points()
        for eq in equilibria:
            axes[1, idx].plot(eq[0], eq[1], 'k*', markersize=12)
        axes[1, idx].set_xlabel('Prey')
        axes[1, idx].set_ylabel('Predators')
        axes[1, idx].grid(True, alpha=0.3)

    plt.suptitle('Effect of Step Size on Euler\'s Method',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()


def print_stability_analysis(model):
    """
    Prints stability analysis results
    """
    print("Stability Analysis")

    results = model.stability_analysis()

    for i, result in enumerate(results, 1):
        print(f"\nEquilibrium Point {i}: {result['equilibrium']}")
        print("-" * 70)
        print("Jacobian Matrix:")
        print(result['jacobian'])
        print(f"\nEigenvalues: {result['eigenvalues']}")
        print(f"Stability: {result['stability']}")

if __name__ == "__main__":
    # Default parameters
    alpha = 1.0 
    beta = 0.1  
    gamma = 1.5 
    delta = 0.075

    model = LotkaVolterraModel(alpha, beta, gamma, delta)

    print_stability_analysis(model)

    t_max = 50.0
    dt = 0.01

    initial_conditions = [
        (10, 5),   
        (20, 5),  
        (30, 10), 
        (15, 15), 
    ]

    trajectories = []
    for x0, y0 in initial_conditions:
        t, x, y = model.euler_method(x0, y0, t_max, dt)
        trajectories.append((t, x, y))

    # Plot 1: Time series for one condition
    t, x, y = trajectories[0]
    plot_time_series(t, x, y)

    # Plot 2: Phase portrait
    plot_phase_portrait(model, trajectories, show_vector_field=True)

    # Plot 3: Compare step sizes
    step_sizes = [0.001, 0.01, 0.1, 0.5]
    compare_step_sizes(model, initial_conditions[2][0],
                      initial_conditions[2][1], t_max, step_sizes)

    plt.show()
