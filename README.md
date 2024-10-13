# Convolutional Neural Network in Rust

A custom implementation of a Convolutional Neural Network (CNN) from scratch in Rust, without relying on high-level machine learning frameworks. This project demonstrates how to build, train, and validate a CNN using Rust's powerful features and libraries.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project showcases a full-fledged implementation of a CNN in Rust, aimed at image classification tasks. It covers the entire pipeline from data loading and preprocessing to model training and validation. The implementation focuses on clarity and educational value, making it a valuable resource for anyone interested in understanding how neural networks work under the hood.

## Features

- **Custom Neural Network Layers**: Implements convolutional, pooling, fully connected, ReLU, and softmax layers from scratch.
- **Data Loading and Preprocessing**: Loads images from a folder structure, converts them to grayscale, and normalizes pixel values.
- **Training and Validation Pipeline**: Includes functions for training the model and validating its performance on a separate dataset.
- **Parallel Computing**: Utilizes Rust's `rayon` crate for parallel computations to speed up training and inference.
- **Custom Optimizer**: Implements the AdamW optimizer for efficient training with weight decay regularization.
- **Progress Monitoring**: Uses `indicatif` to display progress bars during training epochs.
- **Configurable Parameters**: Allows easy adjustment of hyperparameters like learning rate, batch size, and network architecture.

## Getting Started

### Prerequisites

- **Rust Toolchain**: Ensure you have Rust and Cargo installed. If not, install them from [rustup.rs](https://rustup.rs/).

```bash
# Check if Rust is installed
rustc --version
cargo --version
```

- **Image Dataset**: A dataset of images organized in a folder structure where each subfolder represents a class, containing images of that class.

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Build the Project**

   ```bash
   cargo build --release
   ```

   Building in release mode is recommended for performance.

### Data Preparation

1. **Organize Your Dataset**

   Place your dataset in a folder named `data/` at the root of the project. The structure should look like:

   ```
   data/
   ├── class1/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── class2/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   └── ...
   ```

2. **Image Requirements**

   - Images should be in `.jpg`, `.jpeg`, or `.png` format.
   - Images will be converted to grayscale and resized to `64x64` pixels. Ensure your images are at least this size.

## Usage

1. **Run the Program**

   ```bash
   cargo run --release
   ```

   The program will:

   - Load and preprocess the images.
   - Split the data into training and validation sets (default is 90% training, 10% validation).
   - Build the CNN based on the configuration constants.
   - Train the model for the specified number of epochs.
   - Validate the model after each epoch and print the loss and accuracy.

2. **Adjusting Hyperparameters**

   You can modify the hyperparameters by changing the constants in `main.rs`:

   ```rust
   // Training parameters
   const EPOCHS: usize = 10;
   const LEARNING_RATE: f32 = 0.0001;
   const BATCH_SIZE: usize = 32;

   // Model input parameters
   const INPUT_HEIGHT: usize = 64;
   const INPUT_WIDTH: usize = 64;
   const NUM_CLASSES: usize = 10;

   // Convolutional layers configuration
   const CONV_LAYERS_CONFIG: &[(usize, usize)] = &[(16, 3), (32, 3)];

   // Fully connected layers configuration
   const FC_LAYERS_CONFIG: &[usize] = &[256, NUM_CLASSES];
   ```

   - **EPOCHS**: Number of training iterations over the entire dataset.
   - **LEARNING_RATE**: Step size for the optimizer.
   - **BATCH_SIZE**: Number of samples processed before the model is updated.
   - **CONV_LAYERS_CONFIG**: Tuples representing the number of filters and kernel size for each convolutional layer.
   - **FC_LAYERS_CONFIG**: Sizes of the fully connected layers.

3. **Monitoring Training Progress**

   The program displays progress bars for each epoch and prints out the training loss and validation accuracy.

   ```
   Epoch 1 - Loss: 1.2345 [#########################] 100/100 (00:30<00:00)
   Validation Loss: 0.9876, Accuracy: 85.00%
   ```

## Project Structure

- **main.rs**: Entry point of the program. Handles data loading, model creation, training, and validation.
- **data_utils.rs**: Functions for loading and preprocessing data.
- **layers.r**: Contains modules for different neural network layers:
- **optimizers.rs**: Implementation of the AdamW optimizer.
- **Cargo.toml**: Project configuration and dependencies.

## Dependencies

The project uses several crates to facilitate computations:

- [ndarray](https://crates.io/crates/ndarray): For multi-dimensional array manipulations.
- [ndarray-rand](https://crates.io/crates/ndarray-rand): To initialize arrays with random values.
- [rand](https://crates.io/crates/rand): Random number generation.
- [rayon](https://crates.io/crates/rayon): Parallelism for CPU-bound tasks.
- [indicatif](https://crates.io/crates/indicatif): Progress bars for CLI applications.
- [image](https://crates.io/crates/image): Loading and processing images.

These dependencies are specified in the `Cargo.toml` file:

```toml
[dependencies]
ndarray-rand = "0.15.0"
rand = { version = "0.8.5", features = ["std"] }
ndarray = { version = "0.16.1", features = ["rayon"] }
image = { version = "0.25.2", default-features = false, features = ["jpeg"] }
indicatif = "0.17.8"
rayon = "1.10.0"
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Rust Community**: For providing extensive resources and support.
- **Crate Authors**: Thanks to the authors of the crates used in this project for making development easier.
