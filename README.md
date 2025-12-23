# Armocromia Network: Multi-Task Deep Learning for Seasonal Color Analysis

This project explores the application of Convolutional Neural Networks (CNNs) to Seasonal Color Analysis (Armocromia). The system is designed to classify portraits into four primary seasons and six specific subtypes by analyzing the chromatic characteristics of the subject's face.

## Project Overview

The main challenge of this project was to create a model that doesn't just treat labels as independent categories, but follows the logical rules of color theory. For example, in Armocromia, a "Winter" should not be associated with a "Warm" subtype. We tried to replicate how armocromia works in real life; at first the season is evaluated, then the subtype.

To do this, we implemented a Multi-Task CNN that predicts both the season and the subtype. The model uses a Soft Conditioning approach where the prediction of the subtype is influenced by the features learned for the season. Furthermore, we implemented a custom Incompatibility Loss function. This loss uses a binary compatibility matrix to penalize the model whenever it suggests a combination that is theoretically impossible, encouraging the network to learn the underlying rules of the domain.

## Module Breakdown

The repository is organized into specific modules to handle the data pipeline, the model architecture, and the evaluation.

### Architecture and Training
*   **model.py**: Contains the CNNMultiTask class. The architecture uses a shared feature extractor based on convolutional blocks followed by Global Average Pooling (GAP) to reduce the parameter count and prevent overfitting. It then splits into two heads for the different classification tasks.
*   **blocks.py**: A utility script defining the Conv-BatchNorm-ReLU-Pooling sequence used as the fundamental building block for the feature extractor.
*   **compatibility_matrix.py**: This script defines the 4x6 matrix representing valid season-subtype combinations.
*   **train.py**:  Manages the optimization process using the Adam optimizer and a learning rate scheduler. It handles the calculation of the composite loss: Season Cross-Entropy, Subtype Cross-Entropy, and the custom Incompatibility Loss.

### Data Pipeline
*   **preprocess.py**: To prevent the model from being distracted by background colors, we isolate the subject and replace the background with a neutral grey.
*   **split_dataset.py**: Manages the dataset partitioning. It ensures a 80/10/10 split between train, test and validation.
*   **dataset_processing.py**: An orchestrator script that runs the entire preparation flow, from raw data rebalancing to the final image segmentation and cleaning.
*   **labeled_dataset.py and data_loaders.py**: These modules  handle image loading, normalization, and data augmentation (rotations and flips) to improve the model's ability to generalize.

### Evaluation and Predictiona
*   **test.py & main_test.py**: Used to verify model performance on unseen data. 
*   **grad_cam.py**: An interpretability tool that uses Grad-CAM++ to generate heatmaps. This allows for a visual check of which facial areas (skin, eyes, or hair) the model is prioritizing for its decisions.
*   **armocromia_prediction.py**: A script to run the model on new, external images. It applies the full preprocessing pipeline before performing the prediction and saving the results in a text file. To try the model, insert selected pictures in `test_on_our_pictures/our_pics` and run **armocromia_prediction.py**

## Dataset Information

The full dataset is hosted externally due to its size and is required to run the training pipeline. It is available in two formats:

*   **Raw Data**: Original portraits for testing the preprocessing and segmentation pipeline. (https://drive.google.com/drive/folders/1xj7eg08PX0WuXFNhq1o2WVJiB2O8_-LM?usp=drive_link)
*   **Processed Data**: Images already with a grai background, optimized for immediate training. (https://drive.google.com/drive/folders/1r1gVVlLp2MBQnmywI7GisyBj_u788az2?usp=drive_link)

To use the dataset, place the `data_raw` directory or the processed  `data_gray` directory into the project directory.
