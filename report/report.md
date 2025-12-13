# Rastrigin Optimization Report

## Problem Description
The Rastrigin function is a standard benchmark function for testing optimization
algorithms. Due to its large number of local minima, it is challenging for
simple local search methods.

## Algorithms
This project implements and compares the following algorithms:
- Hill Climbing
- Random Restart Hill Climbing
- Simulated Annealing

## Results
Hill Climbing frequently converges to poor local optima.
Random Restart Hill Climbing improves the results by reducing sensitivity to
initial states.
Simulated Annealing achieves near-global optimal solutions when properly
parameterized and evaluated over multiple runs.

## Conclusion
The results show that metaheuristic methods are more robust than simple
heuristic approaches for multimodal optimization problems.
