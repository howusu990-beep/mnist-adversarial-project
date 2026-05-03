
Here’s a clean, professional **README.md** you can paste directly into your GitHub repo:

---

# **Adversarial Attacks and Defenses in Deep Learning (MNIST Case Study)**

## 📌 **Overview**

This project demonstrates how machine learning models can be attacked and defended using adversarial techniques. A neural network is trained on the MNIST dataset, attacked using the Fast Gradient Sign Method (FGSM), and then improved using adversarial training.

---

## 🎯 **Objectives**

* Build a neural network classifier using MNIST
* Apply an adversarial attack (FGSM)
* Measure the drop in model accuracy
* Implement a defense (adversarial training)
* Evaluate improvement after defense

---

## 🛠️ **Technologies Used**

* Python
* PyTorch
* NumPy
* Matplotlib

---

## 📂 **Project Structure**

```
mnist_adversarial_project/
│── data/                 # MNIST dataset
│── train.py              # Model definition and training
│── attack.py             # FGSM attack implementation
│── defense.py            # Adversarial training defense
│── main.py               # Runs full pipeline
│── results.png           # Performance graph
```

---

## 🚀 **How to Run the Project**

### 1. Install dependencies

```bash
pip install torch torchvision numpy matplotlib
```

### 2. Run the project

```bash
python main.py
```

---

## 📊 **Results**

| Stage               | Accuracy |
| ------------------- | -------- |
| Baseline            | 96.74%   |
| Under Attack (FGSM) | 1.33%    |
| After Defense       | 69.25%   |

👉 The FGSM attack causes a significant drop in performance, while adversarial training improves robustness.

---

## 📉 **Visualization**

The results are visualized in `results.png`, showing:

* High baseline accuracy
* Sharp drop under attack
* Partial recovery after defense

---

## ⚔️ **Adversarial Attack (FGSM)**

FGSM perturbs input images using gradients:

[
x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(x, y))
]

This creates small changes that cause the model to misclassify inputs.

---

## 🛡️ **Defense: Adversarial Training**

The model is retrained using adversarial examples, allowing it to:

* Learn more robust patterns
* Reduce sensitivity to small perturbations
* Improve performance under attack

---

## 📈 **Key Findings**

* High accuracy does not mean robustness
* Small perturbations can break models
* Adversarial training significantly improves security
* Defense does not fully restore original performance

---

## 📚 **References**

* Goodfellow et al. (2015) – Adversarial Examples
* Madry et al. (2018) – Adversarial Training
* MNIST Dataset – Yann LeCun

---

## 👤 **Author**

Henrietta Owusu
CTEC 450 – AI Security Project


