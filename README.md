# fMRI Transformer Project

This project is focused on developing a Transformer-based model to classify functional magnetic resonance imaging (fMRI) data. 

The model is designed to work with preprocessed fMRI datasets, leveraging both spatial and temporal features to predict brain activity associated with different tasks or resting states. 

The project includes two primary modes of operation: `train` mode for standard model training and evaluation, and `cross_val` mode for performing k-fold cross-validation.


# Installation
Before running the code, ensure that you have Python 3.8+ installed. 

You will need to install the required Python packages. You can install them using pip:

pip install torch torchmetrics matplotlib seaborn scikit-learn tqdm wandb pandas numpy

## Required Packages:
torch: The PyTorch library for building and training neural networks.

torchmetrics: Metrics for evaluating model performance.

matplotlib: Plotting library for visualizing results.

seaborn: Statistical data visualization.

scikit-learn: Machine learning tools, including K-Fold cross-validation.

tqdm: Progress bar for loops.

wandb: Weights & Biases, a tool for tracking experiments.

pandas: Data manipulation and analysis.

numpy: Fundamental package for numerical computations.


# Prerequisites:
Directory Structure: The cross_val directory should be set up with subject data.

Hyperparameters: Define the hyperparameters, including the number of folds (k), and choose the network (NET_list) and hemisphere (H_list).

Preprocessed Data: Ensure that the data files have been preprocessed and saved as .pkl files.

How to Run:
Ensure that the run_mode is set to 'cross_val' in the main script.

Adjust the hyperparameters and settings as needed.

## Directory Structure

The project expects the dataset to be organized as follows:

```plaintext
root_directory/
├── train/
│   ├── subject_1/
│   │   ├── data_file_1.pkl
│   │   ├── ...
│   ├── subject_2/
│   │   ├── ...
│   └── ...
├── eval/
│   ├── subject_141/
│   │   ├── data_file_1.pkl
│   │   ├── ...
│   └── ...
├── test/
│   ├── subject_159/
│   │   ├── data_file_1.pkl
│   │   ├── ...
│   └── ...
└── cross_val/
    ├── subject_1/
    │   ├── data_file_1.pkl
    │   ├── ...
    └── ...
    ├── subject_176/
    │   ├── ...
    └── ... 
```

## File Naming Conventions:

Each subject folder should contain fMRI data files in .pkl format.

The files should be named according to the hemisphere (LH, RH, or BOTH) and the network (Vis, Default, etc.).

## Hyperparameters

The main script allows for several hyperparameters to be configured. Below is a description of the key hyperparameters:

directory: Path to the root directory containing the train, eval, test, and cross_val folders.

task: The type of task (e.g., 'rest') being classified.

batch_size: Number of samples processed in each batch.

epochs: Number of complete passes through the training dataset.

num_heads: Number of attention heads in the Transformer model.

dropout: Dropout rate applied in various parts of the model to prevent overfitting.

weight_decay: Weight decay (L2 regularization) applied in the optimizer to prevent overfitting.

embedding_dim: Dimension of the embedding space used in the Transformer model.

learning_rate: Learning rate for the optimizer.

NET_list: List of network types (e.g., ['Vis']) used in the dataset.

NET_indexes: List of network indexes corresponding to the NET_list.

H_list: Hemisphere options (e.g., ['RH', 'LH', 'BOTH']) to be used in the analysis.

Avg: Averaging strategy for the data (1 for voxel-level averaging, 2 for TR-level averaging).

Create_synthetic_subjects: Flag to indicate whether to create synthetic subjects (1 for yes, 0 for no).

n_synthetic_subjects: Number of synthetic subjects to generate per original subject.

use_original_for_val_test: Flag to use original data for validation and testing instead of synthetic data.

Group_by_subjects: Flag to indicate whether to group subjects before processing.

group_size: Number of subjects to group together.

slice: Which portion of the fMRI time series to use ('start', 'middle', 'end', 'all').

noise_level: Noise level to add when creating synthetic data.

n_synthetic_TRs: Number of synthetic TRs (Time Repeats) to create.

k: Number of folds to use in cross-validation.

run_mode: Specifies whether to run in 'train' or 'cross_val' mode.

timestamp: Automatically generated timestamp for logging runs.

## Modes of Operation:
### Run Mode: Training

#### Prerequisites:

- The `train`, `eval`, and `test` directories must be properly set up with the subject folders containing the necessary `.pkl` files.
- 140 subjects should be set under 'train/movies' and the same subjects should be set under 'train/rest' (80% of the subjects), 18 in 'eval' (under rest and movies) and 18 in 'test' (under rest and movies)
- Ensure the `exists_list` variable is updated to include all the relevant input files to be processed.

#### Description:

- The `train` mode trains the model using the data in the `train` directory, optionally validating it against the `eval` set and testing it on the `test` set.
- Synthetic subjects can be generated to augment the training data.
- The model and results are logged using Weights and Biases (wandb).

### Run Mode: Cross-Validation

#### Prerequisites:

- The `cross_val` directory must contain all 176 subject folders with the necessary `.pkl` files.
- Ensure the `exists_list` variable includes all the relevant input files.

#### Description:

- The `cross_val` mode performs k-fold cross-validation on the dataset located in the `cross_val` directory.
- The reason for cross validation: grouping into 4-subjects-groups (and then averaging to get the averaged group subject) results with only 44 subjects (176/4). In normal training run_mode we end up with only 4 subjects to test on (10% of 176). With cross_val run mode (with 4 folds and no validation) we test 4 times on 11 different averaged subjects. 
- Each fold trains the model on a subset of the data while validating on the remaining portion.
- The model and results are logged using Weights and Biases (wandb).


# Run the main script:

python main_model.py
