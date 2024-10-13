use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use std::sync::{Arc, Mutex};

use crate::optimizers::AdamWOptimizer;

/// Convolutional layer with learnable filters and biases
pub struct ConvLayer {
    filters: Array4<f32>, // Shape: (num_filters, depth, filter_height, filter_width)
    biases: Array1<f32>,  // Shape: (num_filters)
    first_moment_filters: Array4<f32>,
    second_moment_filters: Array4<f32>,
    first_moment_biases: Array1<f32>,
    second_moment_biases: Array1<f32>,
}

impl ConvLayer {
    /// Creates a new convolutional layer with random initialized filters and zero biases
    pub fn new(num_filters: usize, input_depth: usize, filter_size: usize) -> Self {
        // Initialize filters randomly
        let filters = Array::random(
            (num_filters, input_depth, filter_size, filter_size),
            Uniform::new(-0.1, 0.1),
        );

        // Initialize biases to zero
        let biases = Array::zeros(num_filters);

        // Initialize first and second moments for optimizer to zeros
        let first_moment_filters = Array4::zeros(filters.dim());
        let second_moment_filters = Array4::zeros(filters.dim());
        let first_moment_biases = Array1::zeros(biases.dim());
        let second_moment_biases = Array1::zeros(biases.dim());

        ConvLayer {
            filters,
            biases,
            first_moment_filters,
            second_moment_filters,
            first_moment_biases,
            second_moment_biases,
        }
    }

    /// Forward pass of the convolutional layer
    /// Input shape: (batch_size, depth, height, width)
    /// Output shape: (batch_size, num_filters, out_height, out_width)
    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, _input_depth, input_height, input_width) = input.dim();
        let (num_filters, _filter_depth, filter_height, filter_width) = self.filters.dim();

        let output_height = input_height - filter_height + 1;
        let output_width = input_width - filter_width + 1;

        // Initialize output tensor
        let output_shape = (batch_size, num_filters, output_height, output_width);
        let mut output = Array4::<f32>::zeros(output_shape);

        // Perform convolution operation
        // Parallelize over the batch dimension
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(input.axis_iter(Axis(0)))
            .for_each(|(mut output_sample, input_sample)| {
                for filter_index in 0..num_filters {
                    for i in 0..output_height {
                        for j in 0..output_width {
                            let input_region = input_sample.slice(s![
                                ..,
                                i..i + filter_height,
                                j..j + filter_width
                            ]);
                            let filter = self.filters.slice(s![filter_index, .., .., ..]);
                            let sum = (&input_region * &filter).sum();
                            output_sample[[filter_index, i, j]] = sum + self.biases[filter_index];
                        }
                    }
                }
            });

        output
    }

    /// Backward pass of the convolutional layer
    /// Returns gradients with respect to filters, biases, and input
    pub fn backward(
        &mut self,
        input: &Array4<f32>,
        output_gradient: &Array4<f32>,
    ) -> (Array4<f32>, Array1<f32>, Array4<f32>) {
        let (batch_size, num_filters, output_height, output_width) = output_gradient.dim();
        let (_, input_depth, _, _) = input.dim();
        let (_, _, filter_height, filter_width) = self.filters.dim();

        // Initialize gradients
        let filters_gradient = Arc::new(Mutex::new(Array4::zeros(self.filters.dim())));
        let biases_gradient = Arc::new(Mutex::new(Array1::zeros(self.biases.dim())));
        let input_gradient = Arc::new(Mutex::new(Array4::<f32>::zeros(input.dim())));

        // Parallelize over the batch dimension
        (0..batch_size).into_par_iter().for_each(|batch_index| {
            // Local gradients for this batch sample
            let mut local_filters_gradient = Array4::zeros(self.filters.dim());
            let mut local_biases_gradient = Array1::zeros(self.biases.dim());
            let mut local_input_gradient = Array4::<f32>::zeros(input.dim());

            for filter_index in 0..num_filters {
                for depth_index in 0..input_depth {
                    for i in 0..output_height {
                        for j in 0..output_width {
                            let input_slice = input.slice(s![
                                batch_index,
                                depth_index,
                                i..i + filter_height,
                                j..j + filter_width
                            ]);
                            let grad = output_gradient[[batch_index, filter_index, i, j]];

                            // Update gradients
                            local_filters_gradient
                                .slice_mut(s![filter_index, depth_index, .., ..])
                                .scaled_add(grad, &input_slice);

                            local_input_gradient
                                .slice_mut(s![
                                    batch_index,
                                    depth_index,
                                    i..i + filter_height,
                                    j..j + filter_width
                                ])
                                .scaled_add(
                                    grad,
                                    &self.filters.slice(s![filter_index, depth_index, .., ..]),
                                );
                        }
                    }
                }
                // Update bias gradient
                local_biases_gradient[filter_index] += output_gradient
                    .slice(s![batch_index, filter_index, .., ..])
                    .sum();
            }

            // Accumulate gradients using mutex
            {
                let mut filters_grad_lock = filters_gradient.lock().unwrap();
                *filters_grad_lock += &local_filters_gradient;
            }
            {
                let mut biases_grad_lock = biases_gradient.lock().unwrap();
                *biases_grad_lock += &local_biases_gradient;
            }
            {
                let mut input_grad_lock = input_gradient.lock().unwrap();
                input_grad_lock
                    .slice_mut(s![batch_index, .., .., ..])
                    .assign(&local_input_gradient.slice(s![batch_index, .., .., ..]));
            }
        });

        // Unwrap Arc and Mutex to get the final gradients
        let filters_gradient = Arc::try_unwrap(filters_gradient)
            .expect("Arc still has multiple owners")
            .into_inner()
            .unwrap();
        let biases_gradient = Arc::try_unwrap(biases_gradient)
            .expect("Arc still has multiple owners")
            .into_inner()
            .unwrap();
        let input_gradient = Arc::try_unwrap(input_gradient)
            .expect("Arc still has multiple owners")
            .into_inner()
            .unwrap();

        (filters_gradient, biases_gradient, input_gradient)
    }

    /// Updates the layer's parameters using the optimizer and gradients
    pub fn update(
        &mut self,
        optimizer: &mut AdamWOptimizer,
        filters_gradient: Array4<f32>,
        biases_gradient: Array1<f32>,
    ) {
        optimizer.step(
            &mut self.filters,
            &filters_gradient,
            &mut self.first_moment_filters,
            &mut self.second_moment_filters,
        );
        optimizer.step(
            &mut self.biases,
            &biases_gradient,
            &mut self.first_moment_biases,
            &mut self.second_moment_biases,
        );
    }
}

/// Rectified Linear Unit (ReLU) activation layer
pub struct ReLULayer;

impl ReLULayer {
    /// Forward pass of the ReLU activation function
    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        input.mapv(|x| x.max(0.0))
    }

    /// Backward pass of the ReLU activation function
    /// Computes the gradient of the loss with respect to the input
    pub fn backward(&self, input: &Array4<f32>, output_gradient: &Array4<f32>) -> Array4<f32> {
        let mask = input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        mask * output_gradient
    }
}

/// Max Pooling layer
pub struct MaxPoolLayer {
    pool_size: usize,
}

impl MaxPoolLayer {
    /// Creates a new Max Pooling layer with the specified pool size
    pub fn new(pool_size: usize) -> Self {
        MaxPoolLayer { pool_size }
    }

    /// Forward pass of the Max Pooling layer
    pub fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let (batch_size, depth, height, width) = input.dim();
        let output_height = height / self.pool_size;
        let output_width = width / self.pool_size;

        let output_shape = (batch_size, depth, output_height, output_width);
        let mut output = Array4::<f32>::zeros(output_shape);

        // Perform max pooling operation
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(input.axis_iter(Axis(0)))
            .for_each(|(mut output_sample, input_sample)| {
                for depth_index in 0..depth {
                    for i in 0..output_height {
                        for j in 0..output_width {
                            let region = input_sample.slice(s![
                                depth_index,
                                i * self.pool_size..(i + 1) * self.pool_size,
                                j * self.pool_size..(j + 1) * self.pool_size
                            ]);
                            output_sample[[depth_index, i, j]] =
                                region.iter().cloned().fold(f32::MIN, f32::max);
                        }
                    }
                }
            });

        output
    }

    /// Backward pass of the Max Pooling layer
    /// Computes the gradient of the loss with respect to the input
    pub fn backward(&self, input: &Array4<f32>, output_gradient: &Array4<f32>) -> Array4<f32> {
        let (_, depth, height, width) = input.dim();
        let output_height = height / self.pool_size;
        let output_width = width / self.pool_size;

        let mut input_gradient = Array4::<f32>::zeros(input.dim());

        input_gradient
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(input.axis_iter(Axis(0)))
            .zip(output_gradient.axis_iter(Axis(0)))
            .for_each(
                |((mut input_gradient_sample, input_sample), output_gradient_sample)| {
                    for depth_index in 0..depth {
                        for i in 0..output_height {
                            for j in 0..output_width {
                                let pool_slice = input_sample.slice(s![
                                    depth_index,
                                    i * self.pool_size..(i + 1) * self.pool_size,
                                    j * self.pool_size..(j + 1) * self.pool_size
                                ]);
                                // Find the index of the max value in the pooling window
                                let (max_i, max_j) = pool_slice
                                    .indexed_iter()
                                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                                    .map(|((i, j), _)| (i, j))
                                    .unwrap();
                                input_gradient_sample[[
                                    depth_index,
                                    i * self.pool_size + max_i,
                                    j * self.pool_size + max_j,
                                ]] = output_gradient_sample[[depth_index, i, j]];
                            }
                        }
                    }
                },
            );

        input_gradient
    }
}

/// Fully Connected (Linear) layer
pub struct FCLayer {
    weights: Array2<f32>, // Shape: (output_size, input_size)
    biases: Array1<f32>,  // Shape: (output_size)
    first_moment_weights: Array2<f32>,
    second_moment_weights: Array2<f32>,
    first_moment_biases: Array1<f32>,
    second_moment_biases: Array1<f32>,
}

impl FCLayer {
    /// Creates a new fully connected layer with random initialized weights and zero biases
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Initialize weights randomly
        let weights = Array::random((output_size, input_size), Uniform::new(-0.1, 0.1));

        // Initialize biases to zero
        let biases = Array::zeros(output_size);

        // Initialize first and second moments for optimizer to zeros
        let first_moment_weights = Array2::zeros(weights.dim());
        let second_moment_weights = Array2::zeros(weights.dim());
        let first_moment_biases = Array1::zeros(biases.dim());
        let second_moment_biases = Array1::zeros(biases.dim());

        FCLayer {
            weights,
            biases,
            first_moment_weights,
            second_moment_weights,
            first_moment_biases,
            second_moment_biases,
        }
    }

    /// Forward pass of the fully connected layer
    /// Input shape: (batch_size, input_size)
    /// Output shape: (batch_size, output_size)
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        input.dot(&self.weights.t()) + &self.biases
    }

    /// Backward pass of the fully connected layer
    /// Returns gradients with respect to weights, biases, and input
    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        output_gradient: &Array2<f32>,
    ) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
        // Compute gradients
        let weights_gradient = output_gradient.t().dot(input);
        let biases_gradient = output_gradient.sum_axis(Axis(0));
        let input_gradient = output_gradient.dot(&self.weights);

        (weights_gradient, biases_gradient, input_gradient)
    }

    /// Updates the layer's parameters using the optimizer and gradients
    pub fn update(
        &mut self,
        optimizer: &mut AdamWOptimizer,
        weights_gradient: Array2<f32>,
        biases_gradient: Array1<f32>,
    ) {
        optimizer.step(
            &mut self.weights,
            &weights_gradient,
            &mut self.first_moment_weights,
            &mut self.second_moment_weights,
        );
        optimizer.step(
            &mut self.biases,
            &biases_gradient,
            &mut self.first_moment_biases,
            &mut self.second_moment_biases,
        );
    }
}

/// Softmax activation layer with Cross-Entropy loss
pub struct SoftmaxLayer;

impl SoftmaxLayer {
    /// Forward pass of the softmax activation function
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // For numerical stability, subtract the max value from each row
        let max_values = input.map_axis(Axis(1), |row| row.fold(f32::MIN, |a, &b| a.max(b)));
        let exps = (input - &max_values.insert_axis(Axis(1))).mapv(|x| x.exp());
        let sum_exps = exps.sum_axis(Axis(1)).insert_axis(Axis(1));
        exps / sum_exps
    }

    /// Backward pass of the softmax activation with cross-entropy loss
    /// Computes the gradient of the loss with respect to the input
    pub fn backward(&self, output: &Array2<f32>, targets: &Array1<usize>) -> Array2<f32> {
        let mut gradient = output.clone();
        for (i, &target) in targets.iter().enumerate() {
            gradient[[i, target]] -= 1.0;
        }
        gradient / targets.len() as f32 // Normalize gradient by batch size
    }
}
