# A New Approach for Worm Detection System Based on Neural Network

This repository contains the official Python implementation of the research paper:

> **"A New Approach for Worm Detection System Based on Neural Network"**
>
> **Authors:** Rafik Menassel, Abdeljalil Gattal, and Messai Ahmed
>
> **Published in:** *Indian Journal of Computer Science and Engineering (IJCSE)*, Vol. 14, No. 5, Sep-Oct 2023.
>
> **DOI:** [10.21817/indjcse/2023/v14i5/231405017](https://doi.org/10.21817/indjcse/2023/v14i5/231405017)

## Project Overview

This project implements an anomaly-based **Intrusion/Worm Detection System (IDS)** using a **Multi-Layer Perceptron (MLP)** artificial neural network. Instead of traditional backpropagation gradient-based optimization, the weights and biases of the MLP are trained using metaheuristic optimization algorithms:

1. **Improved Tree-Seed Algorithm (ITSA)** *(Proposed Method)*
2. **Genetic Algorithm (GA)** *(Benchmark)*
3. **Particle Swarm Optimization (PSO)** *(Benchmark)*

The system processes network traffic data from the **NSL-KDD dataset**, transforms multi-class attack labels into binary classifications (`normal` vs. `attack`), normalizes feature values using Min-Max scaling, and compares the training/testing performance of all three metaheuristic approaches across metrics like Mean Squared Error ($MSE$), Accuracy ($ACC$), False Alarm Rate ($FAR$), and Detection Rate ($DR$).

## System Architecture

The model architecture consists of a 3-layer Feedforward Neural Network (3-N-1 MLP):

* **Input Layer:** Variable dimension based on processed NSL-KDD features (after dropping invariant/redundant attributes).
* **Hidden Layer:** 60/79 hidden neurons utilizing Sigmoid activation functions.
* **Output Layer:** Single neuron with Sigmoid activation for binary classification ($[0, 1]$ thresholded at $0.5$).

### Metaheuristic Chromosome Encoding

Each individual/particle/tree in the population represents a flattened vector of floating-point numbers containing:

1. Input-to-Hidden layer weights
2. Hidden layer biases
3. Hidden-to-Output layer weights
4. Output layer bias

$$
\text{Chromosome Dimension} = (N_{\text{input}} \times N_{\text{hidden}}) + N_{\text{hidden}} + (N_{\text{hidden}} \times N_{\text{output}}) + N_{\text{output}}
$$

## Repository Structure

```
.
├── main.py                # Main script to execute training and evaluation
├── functions.py           # Data loading, pre-processing, plotting, and helper utility functions
├── MLP.py                 # Multi-Layer Perceptron neural network implementation
├── TSA_algorithm.py       # Improved Tree-Seed Algorithm (ITSA) optimization implementation
├── GA.py                  # Genetic Algorithm (GA) implementation
├── PSO.py                 # Particle Swarm Optimization (PSO) implementation
└── README.md              # Project documentation
```

## 🛠 Prerequisites & Installation

Ensure you have Python 3.7+ installed along with the required dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

##  How to Run

### 1. Data Preparation

Prepare your training and testing CSV files formatted similarly to the **NSL-KDD** dataset (e.g., `KDDTrain+.txt` and `KDDTest+.txt`).

### 2. Execution

Run the experiment from Python or import the `main` function:

```python
import main

# Run comparison experiment between ITSA, GA, and PSO
main.main('path/to/KDDTrain+.txt', 'path/to/KDDTest+.txt')
```

### 3. Execution Pipeline

1. **Data Pre-processing:** Drops non-informative/redundant features (`garbage`, `num_outbound_cmds`, `is_host_login`), converts multi-class attacks into binary labels (`attack`), label-encodes categorical variables, and scales numerical features to $[0, 1]$.
2. **Metaheuristic Training:** Sequential optimization using **ITSA**, **GA**, and **PSO** for the specified number of epochs.
3. **Testing Evaluation:** Evaluates the best parameter vectors on unseen test data.
4. **Visualization:** Displays comparative convergence curves ($MSE$ vs. Epochs, $ACC$ vs. Epochs) and bar charts for $FAR$, $DR$, and $ACC$.

##  Performance Metrics

The algorithms evaluate performance based on:

* **Accuracy (**$ACC$**):** $\frac{TP + TN}{TP + TN + FP + FN}$
* **Detection Rate (**$DR$**):** $\frac{TP}{TP + FN}$
* **False Alarm Rate (**$FAR$**):** $\frac{FP}{FP + TN}$
* **Mean Squared Error (**$MSE$**):** $\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$

##  Citation

If you use this codebase or research in your work, please cite the original article:

```bibtex
@article{menassel2023worm,
  title={A New Approach for Worm Detection System Based on Neural Network},
  author={Menassel, Rafik and Gattal, Abdeljalil and Ahmed, Messai},
  journal={Indian Journal of Computer Science and Engineering (IJCSE)},
  volume={14},
  number={5},
  pages={729--739},
  year={2023},
  doi={10.21817/indjcse/2023/v14i5/231405017}
}
```