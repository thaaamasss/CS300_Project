import os
import csv
import matplotlib.pyplot as plt

from evaluation.metrics import compute_unlearning_scores, find_best_algorithm
from utils.config import CSV_RESULTS_DIR, PLOTS_DIR

def evaluate_unlearning_algorithms(results, dataset_name):
    # ------------------------------------------------
    # Create results directories
    # ------------------------------------------------

    os.makedirs(CSV_RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ------------------------------------------------
    # Extract metrics
    # ------------------------------------------------

    algorithm_names = []
    remaining_accuracy = []
    deleted_accuracy = []
    time_list = []

    for algo in results:

        algorithm_names.append(algo)

        remaining_accuracy.append(results[algo]["remaining_accuracy"])
        deleted_accuracy.append(results[algo]["deleted_accuracy"])
        time_list.append(results[algo]["time"])

    # ------------------------------------------------
    # Compute weighted scores
    # ------------------------------------------------

    scores = compute_unlearning_scores(
        remaining_accuracy,
        deleted_accuracy,
        time_list
    )

    best_algorithm, best_score = find_best_algorithm(
        algorithm_names,
        scores
    )

    # ------------------------------------------------
    # Save CSV results
    # ------------------------------------------------

    csv_path = os.path.join(
        CSV_RESULTS_DIR,
        f"{dataset_name}_unlearning_results.csv"
    )

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Dataset",
            "Algorithm",
            "Remaining_Accuracy",
            "Deleted_Accuracy",
            "Unlearning_Time",
            "Score"
        ])

        for i in range(len(algorithm_names)):

            writer.writerow([
                dataset_name,
                algorithm_names[i],
                remaining_accuracy[i],
                deleted_accuracy[i],
                time_list[i],
                scores[i]
            ])

    # ------------------------------------------------
    # Plot unlearning time comparison
    # ------------------------------------------------

    plt.figure()

    plt.bar(algorithm_names, time_list)

    plt.title(f"{dataset_name} Unlearning Time Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (seconds)")

    plot_path = os.path.join(
        PLOTS_DIR,
        f"{dataset_name}_unlearning_time.png"
    )

    plt.savefig(plot_path)
    plt.close()

    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print("\nUnlearning Algorithm Evaluation Results")
    print("---------------------------------------")

    for i in range(len(algorithm_names)):

        print(
            "Algorithm:",
            algorithm_names[i],
            "Score:",
            round(scores[i], 4)
        )

    print("\nBest Unlearning Algorithm:", best_algorithm)
    print("Best Score:", round(best_score, 4))

    return best_algorithm, scores

