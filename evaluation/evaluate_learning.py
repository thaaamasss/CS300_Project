import os
import csv
import matplotlib.pyplot as plt

from evaluation.metrics import compute_learning_scores, find_best_algorithm
from utils.config import CSV_RESULTS_DIR, PLOTS_DIR

def evaluate_learning_algorithms(results, dataset_name):
    # ------------------------------------------------
    # Create result directories
    # ------------------------------------------------

    os.makedirs(CSV_RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ------------------------------------------------
    # Extract metrics
    # ------------------------------------------------

    algorithm_names = []
    accuracy_list = []
    time_list = []
    loss_list = []

    for algo in results:

        algorithm_names.append(algo)

        accuracy_list.append(results[algo]["accuracy"])
        time_list.append(results[algo]["time"])
        loss_list.append(results[algo]["loss"])

    # ------------------------------------------------
    # Compute weighted scores
    # ------------------------------------------------

    scores = compute_learning_scores(
        accuracy_list,
        time_list,
        loss_list
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
        f"{dataset_name}_learning_results.csv"
    )

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Dataset",
            "Algorithm",
            "Accuracy",
            "Training_Time",
            "Loss",
            "Score"
        ])

        for i in range(len(algorithm_names)):

            writer.writerow([
                dataset_name,
                algorithm_names[i],
                accuracy_list[i],
                time_list[i],
                loss_list[i],
                scores[i]
            ])

    # ------------------------------------------------
    # Plot learning accuracy
    # ------------------------------------------------

    plt.figure()

    plt.bar(algorithm_names, accuracy_list)

    plt.title(f"{dataset_name} Learning Algorithm Accuracy")
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")

    accuracy_plot_path = os.path.join(
        PLOTS_DIR,
        f"{dataset_name}_learning_accuracy.png"
    )

    plt.savefig(accuracy_plot_path)
    plt.close()

    # ------------------------------------------------
    # Plot training time
    # ------------------------------------------------

    plt.figure()

    plt.bar(algorithm_names, time_list)

    plt.title(f"{dataset_name} Training Time Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Time (seconds)")

    time_plot_path = os.path.join(
        PLOTS_DIR,
        f"{dataset_name}_training_time.png"
    )

    plt.savefig(time_plot_path)
    plt.close()

    # ------------------------------------------------
    # Print results
    # ------------------------------------------------

    print("\nLearning Algorithm Evaluation Results")
    print("-------------------------------------")

    for i in range(len(algorithm_names)):

        print(
            "Algorithm:",
            algorithm_names[i],
            "Score:",
            round(scores[i], 4)
        )

    print("\nBest Learning Algorithm:", best_algorithm)
    print("Best Score:", round(best_score, 4))

    return best_algorithm, scores

