use std::collections::HashMap;
use std::fs;

use image::open;
use ndarray::prelude::*;
use rand::seq::SliceRandom;

/// Loads an image from the specified path and converts it to a normalized Array3<f32>
/// Shape of the output array: (channels, height, width)
pub fn load_image_to_array(path: &str) -> Array3<f32> {
    let img = open(path).expect("Failed to open image");
    let img = img.to_luma8(); // Convert to grayscale
    let (width, height) = img.dimensions();

    // Initialize array to hold image data
    let mut img_array = Array3::<f32>::zeros((1, height as usize, width as usize));

    // Fill the array with normalized pixel values
    for (x, y, pixel) in img.enumerate_pixels() {
        img_array[[0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
    }

    img_array
}

/// Loads image data and labels from a folder structure
/// Expects each subfolder in `folder_path` to represent a class,
/// and contains images belonging to that class.
/// Returns a vector of (image_array, class_index) tuples and a mapping from class names to indices.
pub fn load_data_from_folder(
    folder_path: &str,
) -> (Vec<(Array3<f32>, usize)>, HashMap<String, usize>) {
    let mut data = Vec::new();
    let mut class_to_index = HashMap::new();

    let class_paths = fs::read_dir(folder_path).expect("Could not read directory");

    for (index, class_entry) in class_paths.enumerate() {
        let class_path = class_entry.unwrap().path();
        let class_name = class_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();

        // Map class name to an index
        class_to_index.insert(class_name.clone(), index);

        // Iterate over images in the class folder
        for image_entry in fs::read_dir(class_path).expect("Could not read class folder") {
            let image_path = image_entry.unwrap().path();

            if let Some(extension) = image_path.extension().and_then(|ext| ext.to_str()) {
                if extension.eq_ignore_ascii_case("jpg")
                    || extension.eq_ignore_ascii_case("jpeg")
                    || extension.eq_ignore_ascii_case("png")
                {
                    let image_array = load_image_to_array(image_path.to_str().unwrap());
                    data.push((image_array, index));
                }
            }
        }
    }

    (data, class_to_index)
}

/// Splits the dataset into training and validation sets based on the given split ratio
/// `split_ratio` is the proportion of data to include in the training set
pub fn train_validation_split(
    data: Vec<(Array3<f32>, usize)>,
    split_ratio: f32,
) -> (Vec<(Array3<f32>, usize)>, Vec<(Array3<f32>, usize)>) {
    let mut data = data;
    data.shuffle(&mut rand::thread_rng());

    let split_index = (data.len() as f32 * split_ratio) as usize;
    let (train_data, validation_data) = data.split_at(split_index);
    (train_data.to_vec(), validation_data.to_vec())
}
