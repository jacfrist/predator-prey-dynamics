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
They are:
(0, 0) - extinction
(γ/δ, α/β) - coexistence equilibrium

Stability analysis is then done on these points. 

## Part 3: Implementation and Visualization (The Code)

Euler's method is implemented to show the cycles of the equations and how step size affects the solution.

Visualizations:
Time series plots
Phase portrait
Step size comparison

