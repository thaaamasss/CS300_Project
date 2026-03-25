import torch
import copy
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import load_dataset
from utils.model_saver import save_model
from utils.config import (
    BATCH_SIZE, EPOCHS, DEVICE, RANDOM_SEED
)

from learning_algorithms.sgd_training import sgd_training
from learning_algorithms.adam_training import adam_training
from learning_algorithms.rmsprop_training import rmsprop_training
from learning_algorithms.sisa_training import sisa_training

from deletion_strategies.class_deletion import class_deletion

from unlearning_algorithms.retraining_unlearning import retraining_unlearning
from unlearning_algorithms.finetune_unlearning import finetune_unlearning
from unlearning_algorithms.influence_unlearning import influence_unlearning
from unlearning_algorithms.sisa_unlearning import sisa_unlearning

from evaluation.evaluate_learning import evaluate_learning_algorithms
from evaluation.evaluate_unlearning import evaluate_unlearning_algorithms

from torch.utils.data import DataLoader

torch.manual_seed(RANDOM_SEED)
device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

DATASET_NAME    = 'CIFAR10'
NUM_CLASSES     = 10
INPUT_CHANNELS  = 3
INPUT_SIZE      = 32
CLASS_TO_DELETE = 5    # Dogs


def run_experiment():
    print("=" * 60)
    print("CIFAR-10 EXPERIMENT")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Load dataset via utils/dataset_loader.py
    # ----------------------------------------------------------------
    train_loader, test_loader = load_dataset('cifar10')
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
        print(f"\n[{DATASET_NAME}] Training with {name}...")
        start = time.time()

        model, final_loss = train_fn(
            train_dataset,
            num_classes=NUM_CLASSES,
            input_channels=INPUT_CHANNELS,
            input_size=INPUT_SIZE,
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            device=device
        )
        elapsed = time.time() - start

        model.eval()
        model.to(device)
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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

        save_model(model, DATASET_NAME.lower(), 'learning', f"{name.lower()}_model.pth")

    evaluate_learning_algorithms(learning_results, dataset_name=DATASET_NAME)

    # ----------------------------------------------------------------
    # Select best model
    # ----------------------------------------------------------------
    best_name, best_result = sorted(
        learning_results.items(),
        key=lambda x: (-x[1]['accuracy'], x[1]['time'])
    )[0]
    best_model = best_result['model']
    print(f"\n[{DATASET_NAME}] Best learning algorithm: {best_name} "
          f"(acc={best_result['accuracy']:.4f})")

    # ----------------------------------------------------------------
    # PHASE 2: Class deletion — remove all Dogs (class 5)
    # ----------------------------------------------------------------
    remaining_dataset, deleted_dataset = class_deletion(train_dataset, CLASS_TO_DELETE)
    print(f"[{DATASET_NAME}] Deleted class {CLASS_TO_DELETE} (Dogs). "
          f"Remaining: {len(remaining_dataset)}  Deleted: {len(deleted_dataset)}")

    # ----------------------------------------------------------------
    # PHASE 3: Unlearning — each method on independent deepcopy
    # ----------------------------------------------------------------
    unlearn_algorithms = {
        'Retraining': retraining_unlearning,
        'FineTuning':  finetune_unlearning,
        'Influence':   influence_unlearning,
        'SISA':        sisa_unlearning,
    }
    unlearning_results = {}

    for name, unlearn_fn in unlearn_algorithms.items():
        print(f"\n[{DATASET_NAME}] Unlearning with {name}...")

        # SISA unlearning must run on the SISA-trained model (needs shard_indices)
        # All other methods run on best_model (Adam)
        source_model = learning_results["SISA"]["model"] if name == "SISA" else best_model
        model_copy = copy.deepcopy(source_model)

        start = time.time()
        result_model = unlearn_fn(
            model_copy, remaining_dataset, deleted_dataset,
            num_classes=NUM_CLASSES,
            input_channels=INPUT_CHANNELS,
            input_size=INPUT_SIZE,
            num_epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            device=device
        )
        elapsed = time.time() - start

        result_model.eval()
        result_model.to(device)

        remaining_loader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, shuffle=False)
        deleted_loader   = DataLoader(deleted_dataset,   batch_size=BATCH_SIZE, shuffle=False)

        def get_acc(loader):
            correct = total = 0
            with torch.no_grad():
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = result_model(inputs)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total   += labels.size(0)
            return correct / total

        unlearning_results[name] = {
            'model':              result_model.cpu(),
            'remaining_accuracy': get_acc(remaining_loader),
            'deleted_accuracy':   get_acc(deleted_loader),
            'time':               elapsed,
        }
        print(f"  {name}: remaining={unlearning_results[name]['remaining_accuracy']:.4f}  "
              f"deleted={unlearning_results[name]['deleted_accuracy']:.4f}  "
              f"time={elapsed:.1f}s")

        save_model(result_model, DATASET_NAME.lower(), 'unlearning',
                   f"{name.lower()}_model.pth")

    evaluate_unlearning_algorithms(unlearning_results, dataset_name=DATASET_NAME)


if __name__ == '__main__':
    run_experiment()