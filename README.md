# predator-prey-dynamics
A program that uses Lotka-Volterra equations to model two populations that interact - prey and predators. The visualization shows the cycle of predators eating the prey while the prey population grows naturally. 


## Part 1: The Mathematical Model (The Equations)
The Lotka-Volterra equations are:
dx/dt = αx - βxy  (prey population)
dy/dt = δxy - γy  (predator population)
Where:

x(t) = prey population at time t
y(t) = predator population at time t
α = prey birth rate
β = predation rate (how often predators catch prey)
γ = predator death rate
δ = predator reproduction rate (based on eating prey)

## Part 2: Equilibrium Points & Stability Analysis
Equilibrium points are where populations don't change (dx/dt = 0, dy/dt = 0).
You'd find:

(0, 0) - extinction
(γ/δ, α/β) - coexistence equilibrium

Then do stability analysis:

Compute the Jacobian matrix (partial derivatives of your system)
Find eigenvalues at each equilibrium
Determine if populations return to equilibrium or oscillate

This is where your linear systems knowledge comes in! You'd show that the coexistence equilibrium has purely imaginary eigenvalues, meaning you get periodic orbits (populations cycle forever).

## Part 3: Implementation and Visualization (The Code)

Implement Euler's method to show the cycles of the equations. Show how step size affects the solution.

Visualizations:
Time series plots (populations over time)
Phase portrait - this is the cool part! Plot prey vs predator populations
Show how different initial conditions lead to periodic orbits
Maybe add direction fields/vector fields

