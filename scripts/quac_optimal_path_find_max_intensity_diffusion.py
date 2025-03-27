#! /usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np

from quactography.graph.undirected_graph import Graph
from quactography.adj_matrix.io import load_graph
from quactography.hamiltonian.hamiltonian_qubit_edge import Hamiltonian_qubit_edge
from quactography.solver.qaoa_solver_qu_edge import find_max_cost, multiprocess_qaoa_solver_edge


"""
Tool to run QAOA, optimize parameters, plot cost landscape with optimal
parameters found if only one reps, and returns the optimization results.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "in_graph",
        help="Adjacency matrix which graph we want path that maximizes weights in graph, (npz file)",
        type=str,
    )
    p.add_argument("starting_node",
                   help="Starting node of the graph", type=int)
    p.add_argument("ending_node",
                   help="Ending node of the graph", type=int)
    p.add_argument("output_file",
                   help="Output file name (npz file)", type=str)
    p.add_argument("output_directory",
                    help="Directory where the files will be outputed", type=str,
                    default="data/output_graphs/"
    )
    p.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        help="List of alphas",
        default=[1.2]
    )
    p.add_argument(
        "--reps",
        nargs="+",
        type=int,
        help="List of repetitions to run for the QAOA algorithm",
        default=[1],
    )
    p.add_argument(
        "-npr",
        "--number_processors",
        help="Number of cpu to use for multiprocessing",
        default=1,
        type=int,
    )
    p.add_argument(
        "--optimizer",
        help="Optimizer to use for the QAOA algorithm",
        default="Differential",
        type=str,
    )
    p.add_argument(
        "--plt_cost_landscape",
        help="True or False, Plot 3D and 2D of the cost landscape"
        "(for gamma and beta compact set over all possible angles-0.1 incrementation)",
        action="store_false",
    )
    p.add_argument(
        "--save_only",
        help="Save only the figure without displaying it",
        action="store_true",
    )

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    weighted_graph, _, _ = load_graph(args.in_graph)
    degree = np.count_nonzero(weighted_graph[args.starting_node] != 0)
    graphs = []
    
    if degree <= 3:
        graphs.append(Graph(weighted_graph, args.starting_node, args.ending_node))
    else:
        j = degree%3
        k = np.floor(degree/3)
        for m in range(int(k)):
            l = 0
            xi = 0
            x = 0
            tmp_copy = weighted_graph.copy()
            while xi < len(weighted_graph[args.starting_node])-1:
                while l < 3*m:
                    if tmp_copy[args.starting_node][xi] != 0:
                        tmp_copy[args.starting_node][xi] = 0
                        tmp_copy[xi][args.starting_node] = 0
                        l += 1
                    xi += 1
                while x < 3:
                    if tmp_copy[args.starting_node][xi] != 0:
                        x += 1
                    xi += 1
                tmp_copy[args.starting_node][xi] = 0
                tmp_copy[xi][args.starting_node] = 0
                xi += 1
            graphs.append(Graph(tmp_copy, args.starting_node, args.ending_node))
        
        for i in range(len(weighted_graph[args.starting_node])-1,0,-1):
            while j > 0:
                if weighted_graph[args.starting_node][i] != 0:
                    j -= 1
            weighted_graph[args.starting_node][i] = 0 
            weighted_graph[i][args.starting_node] = 0   
        graphs.append(Graph(weighted_graph, args.starting_node, args.ending_node))
            

    # Construct Hamiltonian when qubits are set as edges,
    # then optimize with QAOA/scipy:
    hamiltonians = []
    output_files = []
    hcount = 0
    for g in graphs:
        hamiltonians.append([Hamiltonian_qubit_edge(g, alpha, hcount) for alpha in args.alphas])
        output_files.append("_" + str(hcount)+ "_" + args.output_file)
        hcount += 1
    # weighted_graph, _, _ = load_graph(args.in_graph)
    # graph = Graph(weighted_graph, args.starting_node, args.ending_node)

    # # Construct Hamiltonian when qubits are set as edges,
    # # then optimize with QAOA/scipy:

    # hamiltonians = [Hamiltonian_qubit_edge(graph, alpha,alpha) for alpha in args.alphas]

    # print(hamiltonians[0].total_hamiltonian.simplify())

    print("\n Calculating qubits as edges......................")
    for i in range(len(args.reps)):
        hcount = 0 
        for hamiltonian in hamiltonians:
            multiprocess_qaoa_solver_edge(
                hamiltonian,
                args.reps[i],
                args.number_processors,
                args.output_file,
                args.output_directory,
                args.optimizer,
                args.plt_cost_landscape,
                args.save_only,
                )
            hcount += 1
        for a in args.alphas:
            find_max_cost(args.output_directory, a, i+1)



if __name__ == "__main__":
    main()
