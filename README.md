# MNIST Adversarial Attack and Defense

## Overview
This project demonstrates how adversarial attacks affect a neural network trained on the MNIST dataset and evaluates a defense mechanism to improve robustness.

## Implementation
The project includes:
- A neural network model built with PyTorch
- FGSM (Fast Gradient Sign Method) adversarial attack
- Adversarial defense through retraining
- Performance evaluation across different conditions

## Results
The model performance shows the impact of adversarial attacks:

- Baseline Accuracy: ~97%
- Accuracy under Attack: ~1%
- Accuracy after Defense: ~65%

These results show that adversarial attacks can significantly degrade model performance, while defense techniques can partially recover accuracy.

## Files
- `main.py` → Model, attack, defense, and evaluation code
- `results.png` → Graph showing performance comparison

## How to Run

1. Install dependencies:
   
