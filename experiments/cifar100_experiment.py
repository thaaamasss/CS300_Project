"""
CIFAR-100 Experiment — Fixed
Fixes applied:
  - Uses load_cifar100() from datasets/cifar100/cifar100_loader.py
  - train_dataset extracted from loader via .dataset
  - Best model selected by training accuracy (accuracy first, time as tiebreaker)
  - Each unlearning method receives an independent deepcopy of best_model
  - loss=0.0 replaced with actual tracked loss (#1)
  - Augmentation added to loader (#8) — see cifar100_loader.py
"""

import torch
import copy
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.cifar100.cifar100_loader import load_cifar100
from learning_algorithms.sgd_training import sgd_training
from learning_algorithms.adam_training import adam_training
from learning_algorithms.rmsprop_training import rmsprop_training
from learning_algorithms.sisa_training import sisa_training
from deletion_strategies.batch_deletion import batch_deletion
from unlearning_algorithms.retraining_unlearning import retraining_unlearning
from unlearning_algorithms.finetune_unlearning import finetune_unlearning
from unlearning_algorithms.influence_unlearning import influence_unlearning
from unlearning_algorithms.sisa_unlearning import sisa_unlearning
from evaluation.evaluate_learning import evaluate_learning_algorithms
from evaluation.evaluate_unlearning import evaluate_unlearning_algorithms
from torch.utils.data import DataLoader

DEVICE              = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE          = 64
NUM_EPOCHS          = 10
DELETION_PERCENTAGE = 10


def run_experiment():
    print("=" * 60)
    print("CIFAR-100 EXPERIMENT")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Load datasets via repo loader
    # ----------------------------------------------------------------
    train_loader, test_loader = load_cifar100(batch_size=BATCH_SIZE)

    train_dataset = train_loader.dataset
    test_dataset  = test_loader.dataset

    # ----------------------------------------------------------------
    # PHASE 1: Train all learning algorithms
    # ----------------------------------------------------------------
    algorithms = {
        'SGD':     sgd_training,
        'Adam':    adam_training,
        'RMSProp': rmsprop_training,
        'SISA':    sisa_training,
    }
    learning_results = {}

    for name, train_fn in algorithms.items():
        print(f"\n[CIFAR-100] Training with {name}...")
        start = time.time()

        model, final_loss = train_fn(
            train_dataset, num_classes=100, in_channels=3,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE
        )
        elapsed = time.time() - start

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

    evaluate_learning_algorithms(learning_results, dataset_name='CIFAR100')

    # ----------------------------------------------------------------
    # Select best model: accuracy first, time as tiebreaker
    # ----------------------------------------------------------------
    best_name, best_result = sorted(
        learning_results.items(),
        key=lambda x: (-x[1]['accuracy'], x[1]['time'])
    )[0]
    best_model = best_result['model']
    print(f"\n[CIFAR-100] Best learning algorithm: {best_name} "
          f"(acc={best_result['accuracy']:.4f})")

    # ----------------------------------------------------------------
    # PHASE 2: Apply batch deletion
    # ----------------------------------------------------------------
    remaining_dataset, deleted_dataset = batch_deletion(
        train_dataset, DELETION_PERCENTAGE
    )
    print(f"[CIFAR-100] Batch deletion {DELETION_PERCENTAGE}%. "
          f"Remaining: {len(remaining_dataset)}  Deleted: {len(deleted_dataset)}")

    # ----------------------------------------------------------------
    # PHASE 3: Each unlearning method on an independent deepcopy
    # ----------------------------------------------------------------
    unlearn_algorithms = {
        'Retraining': retraining_unlearning,
        'FineTuning':  finetune_unlearning,
        'Influence':   influence_unlearning,
        'SISA':        sisa_unlearning,
    }
    unlearning_results = {}

    for name, unlearn_fn in unlearn_algorithms.items():
        print(f"\n[CIFAR-100] Unlearning with {name}...")

        model_copy = copy.deepcopy(best_model)

        start = time.time()
        result_model = unlearn_fn(
            model_copy, remaining_dataset, deleted_dataset,
            num_classes=100, in_channels=3,
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

    evaluate_unlearning_algorithms(unlearning_results, dataset_name='CIFAR100')


if __name__ == '__main__':
    run_experiment()
