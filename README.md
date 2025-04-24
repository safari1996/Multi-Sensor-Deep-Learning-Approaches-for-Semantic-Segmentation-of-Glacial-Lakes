# Multi-Sensor-Deep-Learning-Approaches-for-Semantic-Segmentation-of-Glacial-Lakes
A Comparative Study for Coastal Hydrology Applications
# Image Segmentation Model Comparison

This project provides a TensorFlow-based pipeline for training, evaluating, and comparing different image segmentation models, including U-Net, a simple CNN, and an ASPP-SegNet (Atrous Spatial Pyramid Pooling) model. The code includes data loading, augmentation, training, visualization, and metric plotting functionalities.

## Features

- Custom Metrics: Implements mean IoU and F1 score metrics for model evaluation.
- Data Augmentation: Simple image and mask augmentations to enhance training.
- Multiple Models: Easily compare U-Net, Simple CNN, and ASPP-SegNet architectures.
- Training & Validation: Supports dataset splitting, model training, and saving training histories.
- Visualization: Plots predictions and training/validation curves for F1 and IoU metrics.
- Result Export: Saves training histories as CSV files for analysis.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
- colorama (optional -- for colorizing the console output)

Install dependencies using:

pip install tensorflow numpy matplotlib pandas colorama
## Usage

1. Set Data Paths
   - Place your RGB images and corresponding binary masks in the directories specified by IMAGE_PATH and MASK_PATH in the script.
   - Update these variables at the top of main.py:
     
     IMAGE_PATH = 'path/to/images'
     MASK_PATH = 'path/to/masks'
     
2. Run the Script
   
   python main.py
   
3. Results

- Model training progress will be printed to the console.
- Training history for each model will be saved as CSV files:
  - u-net_history.csv
  - simple_cnn_history.csv
  - aspp_segnet_history.csv
- Plots showing F1 and IoU metrics over epochs will be displayed.
- The script prints the best validation scores for F1, IoU, and loss.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pandas
- colorama

Install dependencies using:

pip install tensorflow numpy matplotlib pandas colorama
## Project Structure

- main.py: Main script for data processing, model definition, training, evaluation, and plotting.
- images/: Directory for input images (user-defined).
- masks/: Directory for binary segmentation masks (user-defined).
- history CSVs: Saved automatically after training.

## Customization

- Adjust the IMAGE_SIZE, BATCH_SIZE, and EPOCHS parameters at the top of the script to fit your dataset or hardware capabilities.
- Implement additional augmentations in the augment function as needed.

## Features

- Custom Metrics: Implements mean IoU and F1 score metrics for model evaluation.
- Data Augmentation: Simple image and mask augmentations to enhance training.
- Multiple Models: Easily compare U-Net, Simple CNN, and ASPP-SegNet architectures.
- Training & Validation: Supports dataset splitting, model training, and saving training histories.
- Visualization: Plots predictions and training/validation curves for F1 and IoU metrics.
- Result Export: Saves training histories as CSV files for analysis.
