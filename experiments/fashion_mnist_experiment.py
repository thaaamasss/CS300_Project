"""
Fashion-MNIST Experiment — Fixed
Fixes applied:
  #1  loss=0.0 replaced with actual tracked loss
  #4  Targeted deletion now semantically targets class 3 (Dress) first 500 samples
  #8  Data augmentation added (train only)
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time, sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.architectures.cnn_model import CNNModel
from learning_algorithms.sgd_training import sgd_training
from learning_algorithms.adam_training import adam_training
from learning_algorithms.rmsprop_training import rmsprop_training
from learning_algorithms.sisa_training import sisa_training
from deletion_strategies.targeted_deletion import targeted_deletion
from unlearning_algorithms.retraining_unlearning import retraining_unlearning
from unlearning_algorithms.finetune_unlearning import finetune_unlearning
from unlearning_algorithms.influence_unlearning import influence_unlearning
from unlearning_algorithms.sisa_unlearning import sisa_unlearning
from evaluation.evaluate_learning import evaluate_learning
from evaluation.evaluate_unlearning import evaluate_unlearning

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
NUM_EPOCHS = 10

# FIX #8: augmentation for training set
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                                    download=True, transform=train_transform)
test_dataset  = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                                    download=True, transform=test_transform)
test_loader   = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# FIX #4: semantically meaningful targeted deletion
# Target class 3 = Dress — simulates deleting all data for a specific garment category
# This is meaningful because we can verify: does the model forget dresses specifically?
targets = torch.tensor(train_dataset.targets)
dress_indices = torch.where(targets == 3)[0].tolist()   # all Dress samples
indices_to_delete = dress_indices[:500]                  # first 500 Dress samples

print(f"[Fashion-MNIST] Targeted deletion: class 3 (Dress), "
      f"{len(indices_to_delete)} samples (indices {indices_to_delete[0]}–{indices_to_delete[-1]})")


def run_learning():
    algorithms = {
        'SGD':     sgd_training,
        'Adam':    adam_training,
        'RMSProp': rmsprop_training,
        'SISA':    sisa_training,
    }
    learning_results = {}

    for name, train_fn in algorithms.items():
        print(f"\n[Fashion-MNIST] Training with {name}...")
        start = time.time()

        # FIX #1: training functions return (model, final_loss)
        model, final_loss = train_fn(
            train_dataset, num_classes=10, in_channels=1,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE
        )
        elapsed = time.time() - start

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total

        learning_results[name] = {
            'model':    model,
            'accuracy': accuracy,
            'time':     elapsed,
            'loss':     final_loss,   # FIX #1
        }
        print(f"  {name}: acc={accuracy:.4f}  loss={final_loss:.4f}  time={elapsed:.1f}s")

    return learning_results


def run_unlearning(best_model, train_dataset):
    # FIX #4: use semantically defined indices_to_delete
    remaining_dataset, deleted_dataset = targeted_deletion(train_dataset, indices_to_delete)

    algorithms = {
        'Retraining': retraining_unlearning,
        'FineTuning':  finetune_unlearning,
        'Influence':   influence_unlearning,
        'SISA':        sisa_unlearning,
    }
    unlearning_results = {}

    for name, unlearn_fn in algorithms.items():
        print(f"\n[Fashion-MNIST] Unlearning with {name}...")
        start = time.time()
        result_model = unlearn_fn(
            best_model, remaining_dataset, deleted_dataset,
            num_classes=10, in_channels=1,
            num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE
        )
        elapsed = time.time() - start

        remaining_loader = DataLoader(remaining_dataset, batch_size=BATCH_SIZE, shuffle=False)
        deleted_loader   = DataLoader(deleted_dataset,  batch_size=BATCH_SIZE, shuffle=False)

        result_model.eval()
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
            'model':              result_model,
            'remaining_accuracy': get_acc(remaining_loader),
            'deleted_accuracy':   get_acc(deleted_loader),
            'time':               elapsed,
        }

    return unlearning_results


if __name__ == '__main__':
    print("=" * 60)
    print("FASHION-MNIST EXPERIMENT")
    print("=" * 60)

    learning_results = run_learning()
    evaluate_learning(learning_results, dataset_name='FashionMNIST')

    best_algo = sorted(
        learning_results.items(),
        key=lambda x: (-x[1]['accuracy'], x[1]['time'])
    )[0]
    print(f"\nBest learning algorithm: {best_algo[0]} ({best_algo[1]['accuracy']:.4f})")

    unlearning_results = run_unlearning(best_algo[1]['model'], train_dataset)
    evaluate_unlearning(unlearning_results, dataset_name='FashionMNIST')
