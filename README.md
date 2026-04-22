# CIFAR-100 Hybrid Architectures

Efficient image classification on CIFAR-100 using hybrid SqueezeNet and ResNet architectures.

## Objective
The primary goal of this project is to explore the synergy between parameter-efficient architectures like **SqueezeNet** and robust feature-learning frameworks like **ResNet**. By hybridizing these designs, we aim to achieve competitive accuracy on the complex CIFAR-100 dataset while maintaining a significantly lower parameter footprint compared to standard deep models.

## Problem it Solves
Standard high-accuracy models (e.g., ResNet-50) are often too parameter-heavy for edge-device deployment, while lightweight models (e.g., SqueezeNet) can struggle with high-dimensional classification tasks like CIFAR-100. This project bridges that gap by implementing a custom hybrid architecture.

## Why it Matters
As deep learning moves toward on-device inference, the need for models that balance **accuracy** and **computational efficiency** becomes critical. This exploration provides insights into how modular components (FireModules and ResidualBlocks) can be combined to optimize this trade-off.

## Constraints
* **Computational Resources**: Optimized for training on T4 GPUs (Colab/Kaggle).
* **Model Size**: Target parameter count is significantly lower than standard ResNet-18/34 implementations.
* **Dataset Complexity**: CIFAR-100's $32 \times 32$ resolution requires careful architectural handling to prevent spatial information loss.

## Approach
* **Baseline Development**: Established initial metrics using standard CNN blocks.
* **Modular Iteration**: Progressively integrated `FireModule` blocks for squeeze/expand operations and `ResidualBlock` for improved gradient flow.
* **Optimization**: Employed `AdamW` optimizer with label smoothing ($0.1$) to enhance generalization.

## Architecture
The project features several architectural iterations:
* **Cifar100v1**: Baseline deep CNN.
* **Cifar100v2**: Pure SqueezeNet-based design using `FireModule`.
* **Cifar100v3**: FireModules with 1x1 projection layers for residual scaling.
* **Cifar100v4**: Pure Residual-based architecture.
* **Cifar100v5 (Hybrid)**: Final model integrating FireModules within a residual framework for optimal efficiency.

## Implementation Notes
* **Framework**: PyTorch-based implementation.
* **Loss Function**: `nn.CrossEntropyLoss(label_smoothing=0.1)`.
* **Optimizer**: `AdamW` with a learning rate of $0.01$.
* **Analysis**: Extensive use of `torchinfo` for parameter auditing and `tqdm` for training visualization.

## Results
| Model | Epochs | Test Accuracy |
| :--- | :---: | :---: |
| **Cifar100v3 (model_2)** | 15 | $45.62\%$ |
| **Cifar100v5 (model_4)** | 11 | $51.71\%$ |

*Note: Latency and GPU memory footprints are significantly reduced through the use of 1x1 squeeze convolutions.*

## Trade-offs
* **Complexity vs. Efficiency**: While FireModules reduce parameters, they introduce architectural complexity in ensuring feature map dimensions match for residual additions.
* **Convergence Speed**: Hybrid models showed faster initial convergence compared to baseline CNNs but required precise hyperparameter tuning.

## Reflection
The integration of residual connections into a lightweight FireModule backbone provides a robust framework for small-image classification. Future work could involve NAS (Neural Architecture Search) to further refine block placement.

## Tech Stack
* **Language**: Python
* **Libraries**: PyTorch, Torchvision, Torchinfo, Matplotlib, NumPy
* **Platform**: Kaggle / Google Colab (GPU accelerated)

## Project Type
AI / Computer Vision
