# MATLAB implementation of the MTS-MARS algorithm

0. Corresponding author: Qi Zhang, Department of Applied Mathematics and Statistics, Stony Brook University, Stony Brook, NY 11794-3600

1. The MTS-MARS algorithm by Qi Zhang and Jiaqiao Hu [1] is implemented for solving single-objective box-constrained expensive stochastic optimization problems.

2. MTS-MARS a random search algorithm for seeking the global optimum of an objective function in a simulation setting. The algorithm can be viewed as an extension of the MARS algorithm proposed in [2] for deterministic optimization, which iteratively finds improved solutions by modifying and sampling from a parameterized probability distribution over the solution space. However, unlike MARS and many other algorithms in this class, which are often population-based, our method only requires a single candidate solution to be generated at each iteration. This is primarily achieved through an effective use of past sampling information by means of embedding multiple nested stochastic approximation type of recursions into the algorithm. We prove the global convergence of the algorithm under general conditions and discuss two special simulation noise cases of interest, in which we show that only one simulation replication run is needed for each sampled solution. A preliminary numerical study is also carried out to illustrate the algorithm.

3. The MTS-MARS algorithm generalizes the TS-MARS algorithm [3] where the former version is designed for sovling single-objective box-constrained expensive deterministic optimization problems.

4. In this implementation, the algorithm samples candidate solutions from a sequence of independent multivariate normal distributions that recursively  approximiates the corresponding Boltzmann distributions [2].

### Reference:
1. Qi Zhang and Jiaqiao Hu: Simulation Optimization Using Multi-time-scale Adaptive Random Search. Submitted to Asia-Pacific Journal of Operational Research, under review.
2. Jiaqiao Hu and Ping Hu (2011): Annealing adaptive search, cross-entropy, and stochastic approximation in global optimization. Naval Research Logistics 58(5):457-477.
3. Qi Zhang and Jiaqiao Hu: A Two-time-scale Adaptive Search Algorithm for Global Optimization. The Proceedings of the 2017 Winter Simulation Conference, pp. 2069-2079.
