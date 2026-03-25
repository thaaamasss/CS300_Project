import argparse
import copy
import csv
import os
import sys
import time
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import load_dataset
from utils.model_saver import load_model
from utils.config import BATCH_SIZE, DEVICE, EPOCHS

from models.architectures.cnn_model import CNNModel
from models.architectures.sisa_model import SISAEnsemble

from unlearning_algorithms.retraining_unlearning import retraining_unlearning
from unlearning_algorithms.finetune_unlearning    import finetune_unlearning
from unlearning_algorithms.influence_unlearning   import influence_unlearning
from unlearning_algorithms.sisa_unlearning        import sisa_unlearning

from utils.config import CSV_RESULTS_DIR, PLOTS_DIR


# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------

FRACTIONS     = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
THRESHOLD_ACC = 0.10    # deleted acc below this = "forgotten"
NUM_SHARDS    = 5       # must match Phase 1 sisa_training NUM_SHARDS

DATASET_META = {
    "mnist": {
        "num_classes": 10, "input_channels": 1, "input_size": 28,
        "class_names": [str(i) for i in range(10)],
    },
    "fashionmnist": {
        "num_classes": 10, "input_channels": 1, "input_size": 28,
        "class_names": ["T-shirt","Trouser","Pullover","Dress","Coat",
                        "Sandal","Shirt","Sneaker","Bag","Boot"],
    },
    "cifar10": {
        "num_classes": 10, "input_channels": 3, "input_size": 32,
        "class_names": ["airplane","automobile","bird","cat","deer",
                        "dog","frog","horse","ship","truck"],
    },
    "cifar100": {
        "num_classes": 100, "input_channels": 3, "input_size": 32,
        "class_names": [str(i) for i in range(100)],
    },
}


# -----------------------------------------------------------------------
# DELETION STRATEGIES
# -----------------------------------------------------------------------

def _get_class_indices(dataset, target_class):
    targets = torch.tensor(dataset.targets)
    return torch.where(targets == target_class)[0].tolist()


def _get_non_class_indices(dataset, target_class):
    targets = torch.tensor(dataset.targets)
    return torch.where(targets != target_class)[0].tolist()


def apply_targeted_deletion(dataset, target_class, fraction):
    """Delete `fraction` of target class samples only."""
    class_indices     = _get_class_indices(dataset, target_class)
    non_class_indices = _get_non_class_indices(dataset, target_class)

    n_delete = int(len(class_indices) * fraction)
    random.shuffle(class_indices)
    deleted_indices    = class_indices[:n_delete]
    kept_class_indices = class_indices[n_delete:]

    remaining_dataset = Subset(dataset, kept_class_indices + non_class_indices)
    deleted_dataset   = Subset(dataset, deleted_indices)
    return remaining_dataset, deleted_dataset


def apply_random_deletion(dataset, target_class, fraction):
    class_indices = _get_class_indices(dataset, target_class)
    n_delete      = int(len(class_indices) * fraction)

    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)
    deleted_indices   = all_indices[:n_delete]
    remaining_indices = all_indices[n_delete:]

    remaining_dataset = Subset(dataset, remaining_indices)
    deleted_dataset   = Subset(dataset, deleted_indices)
    return remaining_dataset, deleted_dataset


def apply_batch_deletion(dataset, target_class, fraction):
    return apply_random_deletion(dataset, target_class, fraction)


STRATEGIES = {
    "Targeted": apply_targeted_deletion,
    "Random":   apply_random_deletion,
    "Batch":    apply_batch_deletion,
}


# -----------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------

def load_best_cnn_model(dataset_key, cfg, device):
    """Load best CNN learning model saved from Phase 1."""
    for algo in ["adam", "sgd", "rmsprop"]:
        try:
            model = CNNModel(
                cfg["input_channels"], cfg["num_classes"], cfg["input_size"]
            )
            model = load_model(
                model, dataset_key, "learning", f"{algo}_model.pth", device
            )
            print(f"  Loaded CNN learning model: {algo}_model.pth")
            return model
        except Exception:
            continue
    raise FileNotFoundError(
        f"No CNN learning model found for {dataset_key}. "
        "Run the Phase 1 experiment first."
    )


def load_sisa_ensemble(dataset_key, cfg, device):
    sisa_path = os.path.join(
        "models", "trained_models", dataset_key, "learning", "sisa_model.pth"
    )
    if not os.path.exists(sisa_path):
        print(f"  WARNING: sisa_model.pth not found at {sisa_path}")
        print(f"  SISA unlearning will be skipped for {dataset_key}.")
        return None

    try:
        # Build dummy ensemble structure to load state dict into
        dummy_shards = [
            CNNModel(cfg["input_channels"], cfg["num_classes"], cfg["input_size"])
            for _ in range(NUM_SHARDS)
        ]
        ensemble = SISAEnsemble(dummy_shards)
        state    = torch.load(sisa_path, map_location="cpu")
        ensemble.load_state_dict(state)

        # Verify shard_indices are present — required for true SISA unlearning
        if ensemble.shard_indices is None:
            print("  WARNING: SISA ensemble has no shard_indices. "
                  "Saved before Phase 1 fix was applied. Skipping SISA.")
            return None

        print(f"  Loaded SISA ensemble: {NUM_SHARDS} shards, "
              f"shard_checkpoints={'yes' if ensemble.shard_checkpoints else 'no'}")
        return ensemble

    except Exception as e:
        print(f"  WARNING: Could not load SISA ensemble — {e}")
        print(f"  SISA unlearning will be skipped.")
        return None


# -----------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------

def evaluate(model, dataset, device, batch_size=64):
    """Accuracy of model on dataset. Works for both CNNModel and SISAEnsemble."""
    model.eval()
    model.to(device)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs        = model(inputs)
            _, predicted   = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)
    model.cpu()
    return correct / total if total > 0 else 0.0


# -----------------------------------------------------------------------
# SINGLE BENCHMARK POINT
# -----------------------------------------------------------------------

def run_single_point(cnn_model, sisa_ensemble,
                     remaining_dataset, deleted_dataset,
                     cfg, device, fast=False):
    results = {}

    # ---- CNN-based algorithms ----
    cnn_fns = {
        "Retraining": retraining_unlearning,
        "FineTuning":  finetune_unlearning,
    }
    if not fast:
        cnn_fns["Influence"] = influence_unlearning

    for algo_name, fn in cnn_fns.items():
        print(f"    [{algo_name}] running...")
        model_copy = copy.deepcopy(cnn_model)
        t0 = time.time()
        try:
            unlearned = fn(
                model_copy,
                remaining_dataset,
                deleted_dataset,
                num_classes=cfg["num_classes"],
                input_channels=cfg["input_channels"],
                input_size=cfg["input_size"],
                num_epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                device=device,
            )
            elapsed  = time.time() - t0
            del_acc  = evaluate(unlearned, deleted_dataset,   device)
            rem_acc  = evaluate(unlearned, remaining_dataset, device)
        except Exception as e:
            print(f"      WARNING: {algo_name} failed — {e}")
            elapsed = time.time() - t0
            del_acc = rem_acc = float("nan")

        results[algo_name] = (del_acc, rem_acc, elapsed)
        print(f"      del={del_acc:.4f}  rem={rem_acc:.4f}  time={elapsed:.1f}s")

    # ---- True SISA unlearning ----
    if sisa_ensemble is not None:
        print(f"    [SISA] running (true shard-aware)...")
        ensemble_copy = copy.deepcopy(sisa_ensemble)
        t0 = time.time()
        try:
            unlearned_ensemble = sisa_unlearning(
                ensemble_copy,
                remaining_dataset,
                deleted_dataset,
                num_classes=cfg["num_classes"],
                input_channels=cfg["input_channels"],
                input_size=cfg["input_size"],
                num_epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                device=device,
            )
            elapsed = time.time() - t0
            del_acc = evaluate(unlearned_ensemble, deleted_dataset,   device)
            rem_acc = evaluate(unlearned_ensemble, remaining_dataset, device)
        except Exception as e:
            print(f"      WARNING: SISA failed — {e}")
            elapsed = time.time() - t0
            del_acc = rem_acc = float("nan")

        results["SISA"] = (del_acc, rem_acc, elapsed)
        print(f"      del={del_acc:.4f}  rem={rem_acc:.4f}  time={elapsed:.1f}s")
    else:
        print(f"    [SISA] skipped — no ensemble loaded")

    return results


# -----------------------------------------------------------------------
# MAIN BENCHMARK RUNNER
# -----------------------------------------------------------------------

def run_benchmark(dataset_key, target_class, fast=False):
    cfg        = DATASET_META[dataset_key]
    device     = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    class_name = cfg["class_names"][target_class]
    tag        = f"{dataset_key}_class{target_class}"

    # Map benchmark dataset key to dataset_loader key
    LOADER_KEY = {
        "fashionmnist": "fashion_mnist",
    }
    loader_key = LOADER_KEY.get(dataset_key, dataset_key)   # ← add this

    # Load dataset
    train_loader, _ = load_dataset(loader_key) 
    print("=" * 65)
    print(f"DELETION STRATEGY BENCHMARK")
    print(f"Dataset     : {dataset_key}")
    print(f"Target class: {target_class} ({class_name})")
    print(f"Fractions   : {[int(f*100) for f in FRACTIONS]}%")
    print(f"Strategies  : {list(STRATEGIES.keys())}")
    print(f"Fast mode   : {fast}")
    print("=" * 65)

    # Load dataset
    loader_key = {"fashionmnist": "fashion_mnist"}.get(dataset_key, dataset_key)
    train_loader, _ = load_dataset(loader_key)
    train_dataset   = train_loader.dataset

    total_class = len(_get_class_indices(train_dataset, target_class))
    print(f"\nTotal samples in class {target_class} ({class_name}): {total_class}")

    # Load both models
    print("\nLoading models...")
    cnn_model     = load_best_cnn_model(dataset_key, cfg, device)
    sisa_ensemble = load_sisa_ensemble(dataset_key, cfg, device)

    has_sisa = sisa_ensemble is not None
    print(f"\nRunning with SISA: {'yes — true shard-aware unlearning' if has_sisa else 'no — sisa_model.pth not found'}")

    # Storage: all_results[strategy][fraction][algo] = (del_acc, rem_acc, time)
    all_results = {s: {} for s in STRATEGIES}

    os.makedirs(CSV_RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---------------------------------------------------------------
    # MAIN SWEEP
    # ---------------------------------------------------------------
    for strategy_name, strategy_fn in STRATEGIES.items():
        print(f"\n{'='*55}")
        print(f"Strategy: {strategy_name}")
        print(f"{'='*55}")

        for fraction in FRACTIONS:
            pct      = int(fraction * 100)
            n_delete = int(total_class * fraction)
            print(f"\n  [{strategy_name}] {pct}% — deleting {n_delete} samples...")

            remaining_dataset, deleted_dataset = strategy_fn(
                train_dataset, target_class, fraction
            )
            print(f"  Remaining: {len(remaining_dataset)}  "
                  f"Deleted: {len(deleted_dataset)}")

            point_results = run_single_point(
                cnn_model, sisa_ensemble,
                remaining_dataset, deleted_dataset,
                cfg, device, fast=fast
            )
            all_results[strategy_name][fraction] = point_results

    # ---------------------------------------------------------------
    # SAVE FULL CSV
    # ---------------------------------------------------------------
    csv_path = os.path.join(CSV_RESULTS_DIR, f"{tag}_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Target_Class", "Strategy", "Fraction_Pct",
            "Unlearning_Algo", "Deleted_Acc", "Remaining_Acc", "Time_s"
        ])
        for strategy_name in all_results:
            for fraction, algo_results in all_results[strategy_name].items():
                for algo_name, (del_acc, rem_acc, elapsed) in algo_results.items():
                    writer.writerow([
                        dataset_key, target_class, strategy_name,
                        int(fraction * 100), algo_name,
                        round(del_acc, 6), round(rem_acc, 6),
                        round(elapsed, 2)
                    ])
    print(f"\nFull results CSV: {csv_path}")

    # ---------------------------------------------------------------
    # COMPUTE + SAVE FORGETTING THRESHOLDS
    # ---------------------------------------------------------------
    algo_names = ["Retraining", "FineTuning"] + \
                 ([] if fast else ["Influence"]) + \
                 (["SISA"] if has_sisa else [])

    thresholds = {s: {} for s in STRATEGIES}
    threshold_csv_path = os.path.join(
        CSV_RESULTS_DIR, f"{tag}_thresholds.csv"
    )

    print(f"\n--- Forgetting thresholds "
          f"(first fraction where deleted acc < {int(THRESHOLD_ACC*100)}%) ---")

    with open(threshold_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Strategy", "Unlearning_Algo",
            "Threshold_Pct", "Deleted_Acc_At_Threshold"
        ])
        for strategy_name in all_results:
            for algo_name in algo_names:
                threshold_frac = None
                threshold_val  = None
                for fraction in FRACTIONS:
                    if fraction not in all_results[strategy_name]:
                        continue
                    if algo_name not in all_results[strategy_name][fraction]:
                        continue
                    da = all_results[strategy_name][fraction][algo_name][0]
                    if da == da and da < THRESHOLD_ACC:   # nan check
                        threshold_frac = int(fraction * 100)
                        threshold_val  = da
                        break

                thresholds[strategy_name][algo_name] = threshold_frac
                label = f"{threshold_frac}%" if threshold_frac else "never"
                print(f"  {strategy_name:10s} | {algo_name:12s}: {label}")
                writer.writerow([
                    strategy_name, algo_name,
                    threshold_frac if threshold_frac else "never",
                    round(threshold_val, 4) if threshold_val else "n/a"
                ])

    print(f"Thresholds CSV: {threshold_csv_path}")

    # ---------------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------------
    _plot_benchmark(
        all_results, thresholds, algo_names,
        dataset_key, target_class, class_name, tag
    )

    print(f"\nBenchmark complete.")
    print(f"  Full CSV  : {csv_path}")
    print(f"  Thresholds: {threshold_csv_path}")
    print(f"  Plot      : results/plots/{tag}_benchmark.png")


# -----------------------------------------------------------------------
# PLOT
# -----------------------------------------------------------------------

def _plot_benchmark(all_results, thresholds, algo_names,
                    dataset_key, target_class, class_name, tag):
    """
    One subplot per unlearning algorithm.
    X axis  = deletion fraction (%).
    Y axis  = deleted accuracy.
    Lines   = one per deletion strategy (Targeted / Random / Batch).
    Dashed horizontal line at THRESHOLD_ACC.
    Vertical lines mark where each strategy first crosses the threshold.
    """
    n_algos = len(algo_names)
    fig, axes = plt.subplots(1, n_algos, figsize=(5 * n_algos, 5), sharey=True)
    if n_algos == 1:
        axes = [axes]

    fractions_pct = [int(f * 100) for f in FRACTIONS]

    strategy_styles = {
        "Targeted": {"color": "#534AB7", "marker": "o", "ls": "-"},
        "Random":   {"color": "#E24B4A", "marker": "s", "ls": "--"},
        "Batch":    {"color": "#EF9F27", "marker": "^", "ls": ":"},
    }

    for ax, algo_name in zip(axes, algo_names):
        for strategy_name, style in strategy_styles.items():
            del_accs = []
            for fraction in FRACTIONS:
                point = (all_results[strategy_name]
                         .get(fraction, {})
                         .get(algo_name, (float("nan"), None, None)))
                da = point[0]
                del_accs.append(da if da == da else None)

            valid_x = [fractions_pct[i] for i, v in enumerate(del_accs)
                       if v is not None]
            valid_y = [v for v in del_accs if v is not None]

            if valid_x:
                ax.plot(valid_x, valid_y,
                        color=style["color"], marker=style["marker"],
                        linestyle=style["ls"], linewidth=2,
                        label=strategy_name)

            # Vertical line at threshold crossing
            t = thresholds.get(strategy_name, {}).get(algo_name)
            if t:
                ax.axvline(t, color=style["color"], linewidth=0.8,
                           linestyle=style["ls"], alpha=0.5)

        # Threshold reference line
        ax.axhline(THRESHOLD_ACC, color="#888780", linewidth=1.2,
                   linestyle="--", label=f"Threshold {int(THRESHOLD_ACC*100)}%")

        ax.set_title(f"{algo_name}", fontsize=11)
        ax.set_xlabel("Deletion fraction (%)")
        if algo_name == algo_names[0]:
            ax.set_ylabel("Deleted accuracy (lower = better forgetting)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(5, 105)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{dataset_key} — deletion strategy benchmark  |  "
        f"target: class {target_class} ({class_name})\n"
        f"Lower curve = better forgetting at that deletion fraction",
        fontsize=10
    )

    plot_path = os.path.join(PLOTS_DIR, f"{tag}_benchmark.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")


# -----------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deletion strategy benchmark — sweep fraction, "
                    "measure forgetting threshold across all unlearning algos"
    )
    parser.add_argument(
        "--dataset", required=True,
        choices=["mnist", "fashionmnist", "cifar10", "cifar100"],
        help="Dataset to benchmark"
    )
    parser.add_argument(
        "--target_class", type=int, required=True,
        help="Class index to use as the deletion target "
             "(e.g. 5 for Dogs in CIFAR-10, 3 for Dress in Fashion-MNIST)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip Influence unlearning (slow CG). "
             "Runs only Retraining + FineTuning + SISA."
    )
    args = parser.parse_args()
    run_benchmark(args.dataset, args.target_class, fast=args.fast)