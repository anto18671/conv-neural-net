// External crate imports
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::prelude::*;
use ndarray::stack;
use rand::seq::SliceRandom;
use rayon::prelude::*;

// Module imports
mod data_utils;
mod layers;
mod optimizers;

// Use items from modules
use data_utils::{load_data_from_folder, train_validation_split};
use layers::{ConvLayer, FCLayer, MaxPoolLayer, ReLULayer, SoftmaxLayer};
use optimizers::AdamWOptimizer;

// Constants for model configuration and training

// Training parameters
const EPOCHS: usize = 10; // Number of training epochs
const LEARNING_RATE: f32 = 0.0001; // Learning rate for optimizer
const BETA1: f32 = 0.9; // Beta1 hyperparameter for AdamW optimizer
const BETA2: f32 = 0.999; // Beta2 hyperparameter for AdamW optimizer
const EPSILON: f32 = 1e-8; // Small epsilon value to prevent division by zero in optimizer
const WEIGHT_DECAY: f32 = 0.01; // Weight decay (L2 regularization) factor

// Model input parameters
const INPUT_CHANNELS: usize = 1; // Number of input channels (e.g., 1 for grayscale images)
const INPUT_HEIGHT: usize = 64; // Height of input images
const INPUT_WIDTH: usize = 64; // Width of input images
const NUM_CLASSES: usize = 10; // Number of output classes

// Pooling parameters
const POOL_SIZE: usize = 2; // Size of the max pooling window

// Convolutional layers configuration
// Each tuple represents (number of filters, kernel size)
// For example, (16, 3) means 16 filters of size 3x3
const CONV_LAYERS_CONFIG: &[(usize, usize)] = &[(16, 3), (32, 3)];

// Fully connected layers configuration
// Each value represents the size of a fully connected layer
const FC_LAYERS_CONFIG: &[usize] = &[256, NUM_CLASSES];

// Batch size for training
const BATCH_SIZE: usize = 32; // Number of samples per training batch

/// Validates the model on the validation dataset
/// Returns the average loss and accuracy on the validation set
fn validate_model(
    validation_data: &[(Array3<f32>, usize)],
    conv_layers: &Vec<ConvLayer>,
    relu_layers: &Vec<ReLULayer>,
    pool_layers: &Vec<MaxPoolLayer>,
    fc_layers: &Vec<FCLayer>,
    softmax_layer: &SoftmaxLayer,
    batch_size: usize,
    flattened_size: usize,
) -> (f32, f32) {
    // Initialize total loss and correct prediction count
    let (total_loss, total_correct_predictions): (f32, usize) = validation_data
        .par_chunks(batch_size)
        .map(|batch| {
            // Prepare batch inputs and targets
            let inputs: Vec<Array3<f32>> = batch.iter().map(|(input, _)| input.clone()).collect();
            let targets: Vec<usize> = batch.iter().map(|(_, target)| *target).collect();

            // Stack inputs into an Array4 (batch_size, channels, height, width)
            let batched_input = stack(
                Axis(0),
                &inputs.iter().map(|x| x.view()).collect::<Vec<_>>(),
            )
            .unwrap();
            let targets = Array1::from(targets);

            // Forward pass through the model
            let mut conv_input = batched_input;

            for ((conv_layer, relu_layer), pool_layer) in conv_layers
                .iter()
                .zip(relu_layers.iter())
                .zip(pool_layers.iter())
            {
                let conv_output = conv_layer.forward(&conv_input);
                let relu_output = relu_layer.forward(&conv_output);
                let pool_output = pool_layer.forward(&relu_output);

                conv_input = pool_output;
            }

            // Flatten the output of the convolutional layers
            let flattened = conv_input
                .into_shape_with_order((batch.len(), flattened_size))
                .expect("Failed to reshape convolutional output");

            // Forward pass through fully connected layers
            let mut fc_input = flattened.clone();

            for fc_layer in fc_layers.iter() {
                fc_input = fc_layer.forward(&fc_input);
            }

            // Apply softmax activation
            let output = softmax_layer.forward(&fc_input);

            // Compute loss and accuracy for the batch
            let mut batch_loss = 0.0;
            let mut batch_correct = 0;

            for (i, &target) in targets.iter().enumerate() {
                batch_loss += -output[[i, target]].ln();
                let predicted_class = output
                    .row(i)
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(index, _)| index)
                    .unwrap();
                if predicted_class == target {
                    batch_correct += 1;
                }
            }

            (batch_loss, batch_correct)
        })
        .reduce(
            || (0.0, 0),
            |(loss1, correct1), (loss2, correct2)| (loss1 + loss2, correct1 + correct2),
        );

    // Compute average loss and accuracy
    let average_loss = total_loss / validation_data.len() as f32;
    let accuracy = total_correct_predictions as f32 / validation_data.len() as f32;

    (average_loss, accuracy)
}

/// Trains the model using the training data and validates using the validation data
fn train_model(train_data: Vec<(Array3<f32>, usize)>, validation_data: Vec<(Array3<f32>, usize)>) {
    // Build Convolutional Layers, Activation Layers, and Pooling Layers
    let mut conv_layers: Vec<ConvLayer> = Vec::new();
    let mut relu_layers: Vec<ReLULayer> = Vec::new();
    let mut pool_layers: Vec<MaxPoolLayer> = Vec::new();

    // Initialize variables for input dimensions
    let mut input_channels = INPUT_CHANNELS;
    let mut current_height = INPUT_HEIGHT;
    let mut current_width = INPUT_WIDTH;

    // Build convolutional layers based on configuration
    for &(num_filters, kernel_size) in CONV_LAYERS_CONFIG.iter() {
        // Create convolutional layer
        conv_layers.push(ConvLayer::new(num_filters, input_channels, kernel_size));
        // Create ReLU activation layer
        relu_layers.push(ReLULayer);
        // Create Max Pooling layer
        pool_layers.push(MaxPoolLayer::new(POOL_SIZE));

        // Update dimensions after convolution
        current_height = current_height - kernel_size + 1;
        current_width = current_width - kernel_size + 1;

        // Update dimensions after pooling
        current_height /= POOL_SIZE;
        current_width /= POOL_SIZE;

        // Update input channels for next layer
        input_channels = num_filters;
    }

    // Compute the size of the flattened feature vector after convolutional layers
    let flattened_size = input_channels * current_height * current_width;

    // Build Fully Connected Layers
    let mut fc_layers: Vec<FCLayer> = Vec::new();
    let mut fc_input_size = flattened_size;
    for &fc_output_size in FC_LAYERS_CONFIG.iter() {
        fc_layers.push(FCLayer::new(fc_input_size, fc_output_size));
        fc_input_size = fc_output_size;
    }

    // Create Softmax Layer
    let softmax_layer = SoftmaxLayer;

    // Initialize the optimizer
    let mut optimizer = AdamWOptimizer::new(LEARNING_RATE, BETA1, BETA2, EPSILON, WEIGHT_DECAY);

    let batch_size = BATCH_SIZE;

    // Training loop
    for epoch in 0..EPOCHS {
        // Create a progress bar for the epoch
        let progress_bar = ProgressBar::new((train_data.len() / batch_size) as u64);
        let progress_style = ProgressStyle::default_bar()
            .template("{msg} [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-");
        progress_bar.set_style(progress_style);

        let mut epoch_loss = 0.0;

        // Shuffle training data at the start of each epoch
        let mut train_data = train_data.clone();
        train_data.shuffle(&mut rand::thread_rng());

        for batch in train_data.chunks(batch_size) {
            // Prepare batch inputs and targets
            let inputs: Vec<Array3<f32>> = batch.iter().map(|(input, _)| input.clone()).collect();
            let targets: Vec<usize> = batch.iter().map(|(_, target)| *target).collect();

            // Stack inputs into an Array4 (batch_size, channels, height, width)
            let batched_input = stack(
                Axis(0),
                &inputs.iter().map(|x| x.view()).collect::<Vec<_>>(),
            )
            .unwrap();

            // Convert targets to Array1
            let targets = Array1::from(targets);

            // Forward pass through convolutional layers
            let mut conv_input = batched_input;
            let mut conv_inputs = Vec::new();
            let mut conv_outputs = Vec::new();
            let mut relu_outputs = Vec::new();

            for ((conv_layer, relu_layer), pool_layer) in conv_layers
                .iter_mut()
                .zip(relu_layers.iter())
                .zip(pool_layers.iter())
            {
                conv_inputs.push(conv_input.clone()); // Store input for backward pass

                let conv_output = conv_layer.forward(&conv_input);
                conv_outputs.push(conv_output.clone());

                let relu_output = relu_layer.forward(&conv_output);
                relu_outputs.push(relu_output.clone());

                let pool_output = pool_layer.forward(&relu_output);

                conv_input = pool_output.clone();
            }

            // Flatten the output of convolutional layers
            let flattened = conv_input
                .into_shape_with_order((batch.len(), flattened_size))
                .expect("Failed to reshape convolutional output");

            // Forward pass through fully connected layers
            let mut fc_input = flattened.clone();
            let mut fc_inputs = Vec::new();
            let mut fc_outputs = Vec::new();

            for fc_layer in fc_layers.iter_mut() {
                fc_inputs.push(fc_input.clone()); // Store input for backward pass
                let fc_output = fc_layer.forward(&fc_input);
                fc_outputs.push(fc_output.clone());
                fc_input = fc_output;
            }

            // Apply softmax activation
            let output = softmax_layer.forward(&fc_input);

            // Compute loss (Cross-Entropy)
            let mut batch_loss = 0.0;
            for (i, &target) in targets.iter().enumerate() {
                batch_loss += -output[[i, target]].ln();
            }
            batch_loss /= batch.len() as f32;
            epoch_loss += batch_loss;

            // Backward pass through softmax layer
            let mut gradient = softmax_layer.backward(&output, &targets);

            // Backward pass through fully connected layers
            for (fc_layer, fc_input) in fc_layers.iter_mut().zip(fc_inputs.iter()).rev() {
                let (weights_gradient, biases_gradient, input_gradient) =
                    fc_layer.backward(&fc_input, &gradient);
                fc_layer.update(
                    &mut optimizer,
                    weights_gradient / batch.len() as f32,
                    biases_gradient / batch.len() as f32,
                );
                gradient = input_gradient;
            }

            // Reshape gradient to match the output shape of the last pooling layer
            let mut gradient = gradient
                .into_shape_with_order((batch.len(), input_channels, current_height, current_width))
                .expect("Failed to reshape gradient");

            // Backward pass through convolutional layers
            for (
                ((conv_layer, relu_layer), pool_layer),
                ((conv_input, conv_output), relu_output),
            ) in conv_layers
                .iter_mut()
                .zip(relu_layers.iter())
                .zip(pool_layers.iter())
                .zip(
                    conv_inputs
                        .iter()
                        .zip(conv_outputs.iter())
                        .zip(relu_outputs.iter()),
                )
                .rev()
            {
                // Backward pass through pooling layer
                let pool_gradient = pool_layer.backward(&relu_output, &gradient);
                // Backward pass through ReLU layer
                let relu_gradient = relu_layer.backward(&conv_output, &pool_gradient);
                // Backward pass through convolutional layer
                let (filters_gradient, biases_gradient, input_gradient) =
                    conv_layer.backward(&conv_input, &relu_gradient);
                conv_layer.update(
                    &mut optimizer,
                    filters_gradient / batch.len() as f32,
                    biases_gradient / batch.len() as f32,
                );
                gradient = input_gradient;
            }

            // Update progress bar
            progress_bar.set_message(format!("Epoch {} - Loss: {:.4}", epoch + 1, batch_loss));
            progress_bar.inc(1);
        }

        let average_epoch_loss = epoch_loss / ((train_data.len() / batch_size) as f32);
        progress_bar.finish_with_message(format!(
            "Epoch {} finished. Average Loss: {:.4}",
            epoch + 1,
            average_epoch_loss
        ));

        // Validation step
        let (validation_loss, accuracy) = validate_model(
            &validation_data,
            &conv_layers,
            &relu_layers,
            &pool_layers,
            &fc_layers,
            &softmax_layer,
            batch_size,
            flattened_size,
        );

        println!(
            "Validation Loss: {:.4}, Accuracy: {:.2}%",
            validation_loss,
            accuracy * 100.0
        );
    }
}

fn main() {
    // Load data from the "data/" folder
    let (dataset, class_to_index) = load_data_from_folder("data/");

    // Print class to index mapping for reference
    println!("Class to index mapping:");
    for (class_name, index) in &class_to_index {
        println!("{}: {}", class_name, index);
    }

    // Split data into training and validation sets (e.g., 90% train, 10% validation)
    let (train_data, validation_data) = train_validation_split(dataset, 0.9);

    println!("Training data size: {}", train_data.len());
    println!("Validation data size: {}", validation_data.len());

    // Train the model
    train_model(train_data, validation_data);
}
