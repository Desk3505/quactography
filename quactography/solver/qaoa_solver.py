from qiskit.primitives import Estimator, Sampler
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
import numpy as np

from quactography.solver.io import save_optimization_results


# Function to find the shortest path in a graph using QAOA algorithm with parallel processing:
def _find_longest_path(args):
    """Summary :  Usage of QAOA algorithm to find the shortest path in a graph.

    Args:
        args (Sparse Pauli list):  Hamiltonian in QUBO representation

    Returns:
        res (minimize):  Results of the minimization
        min_cost (float):  Minimum cost
        alpha_min_cost (list):  List of alpha, minimum cost and binary path
    """
    h = args[0]
    reps = args[1]
    outfile = args[2]

    # Pad with zeros to the left to have the same length as the number of edges:
    for i in range(len(h.exact_path[0])):
        if len(h.exact_path[0]) < h.graph.number_of_edges:
            h.exact_path[i] = h.exact_path[i].zfill(h.graph.number_of_edges + 1)
    print("Path Hamiltonian (quantum reading -> right=q0) : ", h.exact_path)

    # Reverse the binary path to have the same orientation as the classical path:
    h.exact_path_classical_read = [path[::-1] for path in h.exact_path]

    # Create QAOA circuit.
    ansatz = QAOAAnsatz(h.total_hamiltonian, reps, name="QAOA")

    # Plot the circuit layout:
    ansatz.decompose(reps=3).draw(output="mpl", style="iqp")

    # Run on local estimator and sampler. Fix seeds for results reproducibility.
    estimator = Estimator(options={"shots": 1000000, "seed": 42})
    sampler = Sampler(options={"shots": 1000000, "seed": 42})

    # Cost function for the minimizer.
    # Returns the expectation value of circuit with Hamiltonian as an observable.
    def cost_func(params, estimator, ansatz, hamiltonian):
        cost = (
            estimator.run(ansatz, hamiltonian, parameter_values=params)
            .result()
            .values[0]
        )
        return cost

    x0 = np.zeros(ansatz.num_parameters)
    # Minimize the cost function using COBYLA method
    res = minimize(
        cost_func,
        x0,
        args=(estimator, ansatz, h.total_hamiltonian),
        method="COBYLA",
        # callback=callback,
        options={"maxiter": 5000, "disp": False},
        tol=1e-4,
    )

    min_cost = cost_func(res.x, estimator, ansatz, h.total_hamiltonian)
    circ = ansatz.copy()
    circ.measure_all()
    dist = sampler.run(circ, res.x).result().quasi_dists[0]
    dist_binary_probabilities = dist.binary_probabilities()
    save_optimization_results(dist=dist, dist_binary_probabilities=dist_binary_probabilities, min_cost=min_cost, hamiltonian=h, outfile=outfile)  # type: ignore
