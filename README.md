# predator-prey-dynamics
A program that uses Lotka-Volterra equations to model two populations that interact - prey and predators. The visualization shows the cycle of predators eating the prey while the prey population grows naturally. 


## The Mathematical Model
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

## Implementation and Visualization

Euler's method is implemented to show the cycles of the equations and how step size affects the solution.

Visualizations:
Time series plots
Phase portrait
Step size comparison

