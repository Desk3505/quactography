from qiskit.visualization import plot_distribution
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import numpy 
import copy

from quactography.solver.io import load_optimization_results


def plot_distribution_of_probabilities_edge(
    in_file, visu_out_file_total, visu_out_file_selected, save_only
):
    """
    Plot the distribution of probabilities for the optimal path and the selected paths.

    Parameters
    ----------
    in_file: str
        The input file containing the optimization results.
    visu_out_file_total: str
        The output file name for the histogram of all paths.
    visu_out_file_selected: str
        The output file name for the histogram of selected paths.
    save_only: bool
        If True, the figure is saved without displaying it.

    Returns
    -------
    None
    """

    _, dist_binary_prob, min_cost, h, _, _, _ = load_optimization_results(
        in_file
    )
    # convert dist_binary_prob a dictionary
    dist_binary_prob = dist_binary_prob.item()
    min_cost = min_cost.item()
    h = h.item()
    exact_path = (h.exact_path[0][::-1]).zfill(len(next(iter(dist_binary_prob))))
    count1 = []
    count2 = []
    colors = []
    count3 = copy.copy(dist_binary_prob)
    count3.pop(exact_path)

    for path, prob in dist_binary_prob.items():
        count1.append(path)
        count2.append(prob)
        if path == exact_path:
            colors.append('tab:green')
        else:
            if prob < dist_binary_prob[exact_path]:
                colors.append("tab:gray")
            else:   
                colors.append("tab:red")
        
    # Plot distribution of probabilities:
    bar_container = plt.bar(list(count1),list(count2),color=colors)
    plt.tick_params("x",rotation=90)
    plt.xticks((numpy.arange(len(count1))),count1)
    plt.bar_label(bar_container, fmt='%.2f', fontsize=8, padding=3)

 
    if not save_only:
        plt.show()
    # Save plot of distribution:
    plt.savefig(visu_out_file_total)
    print("Distribution of probabilities saved in ", visu_out_file_total)

    # print(max(dist_binary_prob, key=dist_binary_prob.get))
    # bin_str = list(map(int, max(dist_binary_prob, key=dist_binary_prob.get)))
    # bin_str_reversed = bin_str[::-1]
    # bin_str_reversed = numpy.array(bin_str_reversed)

    # Check if optimal path in a subset of most probable paths:
    sorted(
        dist_binary_prob, key=dist_binary_prob.get, reverse=True
    )

    # Dictionary keys and values where key = binary path, value = probability:
    # Find maximal probability in all values of the dictionary:
    max_probability = max(dist_binary_prob.values())
    selected_paths = []
    for path, probability in dist_binary_prob.items():

        probability = probability / max_probability
        dist_binary_prob[path] = probability
        # print(
        #     f"Path (quantum read -> right=q0): {path} with ratio proba/max_proba : {probability}"
        # )

        percentage = 0.5
        # Select paths with probability higher than percentage of the maximal probability:
        if probability > percentage:
            selected_paths.extend([path, probability])

    # Sort the selected paths by probability from most probable to least probable:
    selected_paths = sorted(
        selected_paths[::2], key=lambda x: dist_binary_prob[x], reverse=True
    )

    

    # match_found = False
    # for i in selected_paths:
    #     if i in h.exact_path:
    #         match_found = True
    #         break

    plot_distribution(
        {key: dist_binary_prob[key] for key in selected_paths},
        figsize=(16, 14),
        title=("Distribution of probabilities for selected paths"),
        # \n Right path FOUND (quantum read): {h.exact_path}"
        # if match_found
        # else f"Distribution of probabilities for selected paths
        # \n Right path NOT FOUND (quantum read): {h.exact_path}"
        color="pink",  # if match_found else "lightblue",
        sort="value_desc",
        filename=visu_out_file_selected,
    )
    print("Distribution of probabilities for selected paths saved in ", visu_out_file_selected)
    if not save_only:
        plt.show()
    # target_string=h.exact_path,)
    # if match_found:
    #     print(
    #         "The optimal solution is in the subset of solutions found by QAOA.\n______")

    # else:
    #     print(
    #         "The solution is not in given subset of solutions found by QAOA.\n____")
    #     )

 
def plot_distribution_comparison(
    in_folder, visu_out_file_selected, save_only
):
    """
    Plot the distribution of probabilities for the optimal path
    and the selected paths.

    Parameters
    ----------
    in_folder: str
        The folder containing the optimization results files.
    visu_out_file_selected: str
        The output file name for the histogram of selected paths.
    save_only: bool
        If True, the figure is saved without displaying it.

    Returns
    -------
    None
    """
    counts = []
    sumDir = 0
    path = Path(in_folder)

    glob_path = path.glob('*.npz')

    for in_file_path in glob_path:
        _, dist_binary_prob, min_cost, h, _, _, _ = load_optimization_results(
            in_file_path
        )
        dist_binary_prob = dist_binary_prob.item()
        min_cost = min_cost.item()
        h = h.item()
        sorted(
            dist_binary_prob, key=dist_binary_prob.get, reverse=True
        )

        max_probability = max(dist_binary_prob.values())
        selected_paths = []
        count = []

        for path, probability in dist_binary_prob.items():

            probability = probability / max_probability
            dist_binary_prob[path] = probability

            percentage = 0.7
            # Select paths with probability higher than percentage of the maximal probability:
            if probability > percentage:
                selected_paths.extend([path, probability])

        # Sort the selected paths by probability from most probable to least probable:
        selected_paths = sorted(
            selected_paths[::2], key=lambda x: dist_binary_prob[x], reverse=True
            )
        count.append({key: dist_binary_prob[key] for key in selected_paths})
        counts.append(count)
        sumDir += 1

    # match_found = False
    # for i in selected_paths:
    #     if i in h.exact_path:
    #         match_found = True
    #         break

    plots = []
    legend = []
    colors = []
    color = iter(cm.rainbow(numpy.linspace(0, 1, sumDir)))
    for j in range(sumDir):
        plots.append(counts[j][0])
        legend.append("File_" + str(j+1))
        colors.append(next(color))

    plot_distribution(
        plots,
        figsize=(16, 14),
        title=("Distribution of probabilities for selected paths"),
        # \n Right path FOUND (quantum read): {h.exact_path}"
        # if match_found
        # else f"Distribution of probabilities for selected paths
        # \n Right path NOT FOUND (quantum read): {h.exact_path}"
        color=colors,  # if match_found else "lightblue",
        sort="value_desc",
        legend=legend,
        filename=visu_out_file_selected,
    )
    if not save_only:
        plt.show()
