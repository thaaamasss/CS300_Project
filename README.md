# Machine Unlearning Experimental Framework (Work in Progress)

## Overview

Machine learning models often retain information from the datasets used during training. When data must be removed due to privacy concerns, simply deleting the data from the dataset is not sufficient because the trained model may still remember it.

This project focuses on implementing and evaluating **machine unlearning techniques** that allow a trained model to remove the influence of specific training samples without retraining the entire model from scratch.

**Note:**
The implementation of the learning and unlearning framework is complete, but experiments and result analysis are still ongoing. This repository currently contains the core system implementation.

---

# Project Objectives

The main objectives of this project are:

* Implement multiple **learning algorithms** for training models
* Implement different **machine unlearning techniques**
* Apply deletion strategies to remove selected training samples
* Evaluate the effectiveness of unlearning algorithms
* Compare learning and unlearning performance across datasets

At the current stage, the focus is on building the **experimental framework** that supports these evaluations.

---

# Datasets

The framework supports the following datasets:

### MNIST

* Handwritten digit dataset
* 10 classes
* Used for initial experimentation due to its simplicity

### Fashion-MNIST

* Clothing item dataset
* Similar structure to MNIST but visually more complex

### CIFAR-10

* 60,000 color images
* 10 object categories
* Used for more challenging image classification tasks

### CIFAR-100

* 100 object classes
* Used to test scalability of learning and unlearning algorithms

---

# Learning Algorithms Implemented

The following training methods are implemented in the framework.

### SGD (Stochastic Gradient Descent)

Basic optimization algorithm widely used in deep learning.

### Adam

Adaptive optimization algorithm that adjusts learning rates for each parameter.

### RMSProp

An adaptive learning rate method designed for stable gradient updates.

### SISA Training

Sharded, Isolated, Sliced and Aggregated training approach designed to enable efficient machine unlearning.

---

# Machine Unlearning Algorithms Implemented

The framework includes multiple strategies to remove training data influence.

### Retraining-Based Unlearning

Retrains the model from scratch after removing selected training samples.

### Fine-Tuning Unlearning

Partially retrains the model to reduce the influence of deleted data.

### SISA Unlearning

Retrains only the affected shards of the dataset, significantly reducing computation cost.

### Influence-Based Unlearning

Attempts to remove the effect of deleted samples by adjusting model parameters.

---

# Deletion Strategies

Different strategies are implemented to select data samples for removal.

* **Random Deletion** – Randomly selects training samples to delete
* **Class Deletion** – Removes all samples from a specific class
* **Targeted Deletion** – Deletes specific samples based on indices
* **Batch Deletion** – Removes groups of samples simultaneously

---

# Running the Framework

Example command to run experiments for a dataset:

```
python -m experiments.mnist_experiment
```

or

```
python -m experiments.cifar10_experiment
```

These scripts execute the following pipeline:

1. Load dataset
2. Train models using different learning algorithms
3. Apply deletion strategies
4. Perform machine unlearning
5. Store experimental results

---

# Planned Evaluation

The final experiments will compare:

* Learning algorithm performance
* Unlearning effectiveness
* Model accuracy before and after unlearning
* Training and unlearning time

Results and visualizations will be stored in the `results/` directory.

---

# Future Work

The next steps for the project include:

* Running full experiments across all datasets
* Analyzing learning and unlearning performance
* Generating result visualizations
* Adding privacy verification methods such as membership inference attacks

