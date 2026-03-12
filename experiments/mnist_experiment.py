"""
MNIST Experiment — Fixed
Fixes applied:
  - Uses load_mnist() from datasets/mnist/mnist_loader.py (no manual download)
  - train_dataset extracted from loader via .dataset
  - Best model selected by training accuracy (accuracy first, time as tiebreaker)
  - Each unlearning method receives an independent deepcopy of best_model
  - loss=0.0 replaced with actual tracked loss (#1)
  - Augmentation added to loader (#8) — see mnist_loader.py
"""

import torch
import copy
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.mnist.mnist_loader import load_mnist
from learning_algorithms.sgd_training import sgd_training
from learning_algorithms.adam_training import adam_training
from learning_algorithms.rmsprop_training import rmsprop_training
from learning_algorithms.sisa_training import sisa_training
from deletion_strategies.random_deletion import random_deletion
from unlearning_algorithms.retraining_unlearning import retraining_unlearning
from unlearning_algorithms.finetune_unlearning import finetune_unlearning
from unlearning_algorithms.influence_unlearning import influence_unlearning
from unlearning_algorithms.sisa_unlearning import sisa_unlearning
from evaluation.evaluate_learning import evaluate_learning_algorithms as evaluate_learning
from evaluation.evaluate_unlearning import evaluate_unlearning_algorithms as evaluate_unlearning
from torch.utils.data import DataLoader

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_EPOCHS = 10
DELETE_SAMPLES = 500


def run_experiment():
    print("=" * 60)
    print("MNIST EXPERIMENT")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Load datasets via repo loader
    # ----------------------------------------------------------------
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)

    # Extract underlying dataset objects for deletion strategies
    # (deletion strategies need Subset, not DataLoader)
    train_dataset = train_loader.dataset
    test_dataset  = test_loader.dataset

    # ----------------------------------------------------------------
    # PHASE 1: Train all learning algorithms, collect (model, metrics)
    # ----------------------------------------------------------------
    algorithms = {
        'SGD':     sgd_training,
        'Adam':    adam_training,
        'RMSProp': rmsprop_training,
        'SISA':    sisa_training,
    }
    learning_results = {}

    for name, train_fn in algorithms.items():
        print(f"\n[MNIST] Training with {name}...")
        start = time.time()

        # Returns (model, final_loss) — see fixed training files
        model, final_loss = train_fn(
            train_dataset, num_classes=10, input_channels=1, input_size=28,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE
        )
        elapsed = time.time() - start

        # Evaluate on test set
        model.eval()
        model.to(DEVICE)
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += labels.size(0)
        accuracy = correct / total

        learning_results[name] = {
            'model':    model.cpu(),
            'accuracy': accuracy,
            'time':     elapsed,
            'loss':     final_loss,
        }
        print(f"  {name}: acc={accuracy:.4f}  loss={final_loss:.4f}  time={elapsed:.1f}s")

    evaluate_learning(learning_results, dataset_name='MNIST')

    # ----------------------------------------------------------------
    # Select best model: accuracy first, time as tiebreaker
    # ----------------------------------------------------------------
    best_name, best_result = sorted(
        learning_results.items(),
        key=lambda x: (-x[1]['accuracy'], x[1]['time'])
    )[0]
    best_model = best_result['model']
    print(f"\n[MNIST] Best learning algorithm: {best_name} "
          f"(acc={best_result['accuracy']:.4f})")

    # ----------------------------------------------------------------
    # PHASE 2: Apply deletion strategy
    # ----------------------------------------------------------------
    deleted_indices, remaining_dataset, deleted_dataset = random_deletion(
        train_dataset, DELETE_SAMPLES
    )
    print(f"[MNIST] Deleted {len(deleted_indices)} random samples. "
          f"Remaining: {len(remaining_dataset)}")

    # ----------------------------------------------------------------
    # PHASE 3: Apply each unlearning method to an independent deepcopy
    # ----------------------------------------------------------------
    unlearn_algorithms = {
        'Retraining': retraining_unlearning,
        'FineTuning':  finetune_unlearning,
        'Influence':   influence_unlearning,
        'SISA':        sisa_unlearning,
    }
    unlearning_results = {}

    for name, unlearn_fn in unlearn_algorithms.items():
        print(f"\n[MNIST] Unlearning with {name}...")

        # Each method gets its own independent deepcopy — no state bleeds across methods
        model_copy = copy.deepcopy(best_model)

        start = time.time()
        result_model = unlearn_fn(
            model_copy, remaining_dataset, deleted_dataset,
            num_classes=10, input_channels=1, input_size=28,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE
        )
        elapsed = time.time() - start

        result_model.eval()
        result_model.to(DEVICE)

        remaining_loader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, shuffle=False)
        deleted_loader   = DataLoader(deleted_dataset,   batch_size=BATCH_SIZE, shuffle=False)

        def get_acc(loader):
            correct = total = 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = result_model(inputs)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total   += labels.size(0)
            return correct / total

        unlearning_results[name] = {
            'model':               result_model.cpu(),
            'remaining_accuracy':  get_acc(remaining_loader),
            'deleted_accuracy':    get_acc(deleted_loader),
            'time':                elapsed,
        }
        print(f"  {name}: remaining={unlearning_results[name]['remaining_accuracy']:.4f}  "
              f"deleted={unlearning_results[name]['deleted_accuracy']:.4f}  "
              f"time={elapsed:.1f}s")

    evaluate_unlearning(unlearning_results, dataset_name='MNIST')


if __name__ == '__main__':
    run_experiment()
