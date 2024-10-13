use ndarray::azip;
use ndarray::{ArrayBase, Dimension};

/// AdamW Optimizer implementation
pub struct AdamWOptimizer {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub time_step: usize,
}

impl AdamWOptimizer {
    /// Creates a new AdamW optimizer with the given hyperparameters
    pub fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        AdamWOptimizer {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            time_step: 0,
        }
    }

    /// Performs a single optimization step for a parameter tensor
    pub fn step<D, Dim>(
        &mut self,
        parameter: &mut ArrayBase<D, Dim>,
        gradient: &ArrayBase<D, Dim>,
        first_moment: &mut ArrayBase<D, Dim>,
        second_moment: &mut ArrayBase<D, Dim>,
    ) where
        D: ndarray::DataMut<Elem = f32>,
        Dim: Dimension,
    {
        self.time_step += 1;
        let lr_t = self.learning_rate * (1.0 - self.beta2.powi(self.time_step as i32)).sqrt()
            / (1.0 - self.beta1.powi(self.time_step as i32));

        // Update biased first moment estimate
        first_moment.zip_mut_with(gradient, |m, &g| {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
        });

        // Update biased second raw moment estimate
        second_moment.zip_mut_with(gradient, |v, &g| {
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
        });

        let beta1_correction = 1.0 - self.beta1.powi(self.time_step as i32);
        let beta2_correction = 1.0 - self.beta2.powi(self.time_step as i32);

        // Update parameters
        azip!((param in parameter, m in first_moment, v in second_moment) {
            let m_hat = *m / beta1_correction;
            let v_hat = *v / beta2_correction;
            *param -= lr_t * (m_hat / (v_hat.sqrt() + self.epsilon) + self.weight_decay * *param);
        });
    }
}
