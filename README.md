# fMRI Transformer Project

This project is focused on developing a Transformer-based model to classify functional magnetic resonance imaging (fMRI) data. The model is designed to work with preprocessed fMRI datasets, leveraging both spatial and temporal features to predict brain activity associated with different tasks or resting states. The project includes two primary modes of operation: `train` mode for standard model training and evaluation, and `cross_val` mode for performing k-fold cross-validation.

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

Hyperparameters
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



Installation
Before running the code, ensure that you have Python 3.8+ installed. You will need to install the required Python packages. You can install them using pip:

pip install torch torchmetrics matplotlib seaborn scikit-learn tqdm wandb pandas numpy

Required Packages:
torch: The PyTorch library for building and training neural networks.
torchmetrics: Metrics for evaluating model performance.
matplotlib: Plotting library for visualizing results.
seaborn: Statistical data visualization.
scikit-learn: Machine learning tools, including K-Fold cross-validation.
tqdm: Progress bar for loops.
wandb: Weights & Biases, a tool for tracking experiments.
pandas: Data manipulation and analysis.
numpy: Fundamental package for numerical computations.

File Naming Conventions
Each subject folder should contain fMRI data files in .pkl format.
The files should be named according to the hemisphere (LH, RH, or BOTH) and the network (Vis, Default, etc.).
3. Modes of Operation
A. run_mode='train'
Description:
This mode trains the model on the training dataset and evaluates it on a separate evaluation dataset. After training, the model is tested on a test dataset. This mode is intended for standard training and validation of the model.

Prerequisites:
Directory Structure: The train, eval, and test directories should be properly set up with subject data as described above.
Hyperparameters: Define the hyperparameters (batch size, epochs, learning rate, etc.) and choose the network (NET_list) and hemisphere (H_list).
Preprocessed Data: Ensure that the data files have been preprocessed and saved as .pkl files.
How to Run:
Ensure that the run_mode is set to 'train' in the main script.

Adjust the hyperparameters and settings as needed.

Run the main script:

python main_model.py

Expected Output:
The model will be trained on the training dataset.
Validation will occur after each epoch using the evaluation dataset.
The best model (based on evaluation loss) will be saved.
Finally, the model will be tested on the test dataset, and the results will be logged.
B. run_mode='cross_val'
Description:
This mode performs k-fold cross-validation on the dataset. The data is split into k folds, where the model is trained on k-1 folds and tested on the remaining fold. This process is repeated k times, with each fold being used once as the test set.

Prerequisites:
Directory Structure: The cross_val directory should be set up with subject data.
Hyperparameters: Define the hyperparameters, including the number of folds (k), and choose the network (NET_list) and hemisphere (H_list).
Preprocessed Data: Ensure that the data files have been preprocessed and saved as .pkl files.
How to Run:
Ensure that the run_mode is set to 'cross_val' in the main script.

Adjust the hyperparameters and settings as needed.

Run the main script:

bash
Copy code
python main_model.py
Expected Output:
The model will be trained and tested on k different folds.
The average accuracy across all folds will be reported.
Each fold's best model will be saved separately.
4. Hyperparameters and Configuration
NET_list: List of networks to use in the training (e.g., ['Vis']).
H_list: List of hemispheres to use (e.g., ['RH', 'LH']).
batch_size: Number of samples per batch.
epochs: Number of training epochs.
learning_rate: Learning rate for the optimizer.
dropout: Dropout rate for regularization.
k: Number of folds for cross-validation (only for cross_val mode).
5. Notes
Synthetic Data: The script includes options for creating synthetic subjects by adding Gaussian noise to the original data. This can be controlled with the Create_synthetic_subjects and n_synthetic_subjects parameters.
Logging: All training and evaluation metrics are logged using Weights & Biases (wandb). Make sure to log in to wandb before running the scripts.
