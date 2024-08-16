import glob
import pickle
import pandas as pd
import torch
import os.path as osp
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from sklearn.decomposition import PCA

# Function to perform z-score normalization
def z_score_normalize(data, axis):
    return (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)

# Function to perform z-score normalization for series data
def z_score_normalize_series(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Function to add Gaussian noise to the data
def add_gaussian_noise(data, noise_level=0.01, n_synthetic=3):
    synthetic_samples = []
    for _ in range(n_synthetic):
        noise = np.random.randn(*data.shape) * noise_level
        synthetic_samples.append(data + noise)
    return synthetic_samples

# Function to average voxel data from multiple files
def average_voxels(data_files):
    averaged_voxs = []
    metadata = None

    for file_path in data_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, pd.DataFrame):
                voxel_data = data.iloc[:, :-3]
                mean_voxels = voxel_data.mean(axis=1)
                normalized_mean_voxels = z_score_normalize_series(mean_voxels)
                averaged_voxs.append(normalized_mean_voxels)
                metadata = data.iloc[:, -3:]
            else:
                raise TypeError("Loaded data is not a DataFrame. Check the file format.")

    if not averaged_voxs:
        return None

    averaged_data = pd.concat(averaged_voxs, axis=1)
    final_data = pd.concat([averaged_data, metadata.reset_index(drop=True)], axis=1)

    return final_data

# Function to average the last few TRs (time points) for each movie and add synthetic samples
def average_TRs(data_files, noise_level=0.01, n_synthetic_TRs=3):   # for PCA test use n_components=10
    averaged_trs = {i: [] for i in range(15)}
    metadata_list = {i: [] for i in range(15)}

    for file_path in data_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, pd.DataFrame):
                for y_value in range(15):
                    movie_data = data[data['y'] == y_value]
                    voxel_data = movie_data.iloc[:, :-3]
                    metadata = movie_data.iloc[:, -3:]

                    last_TRs = voxel_data.iloc[-5:, :]
                    mean_TRs = last_TRs.mean(axis=0)
                    averaged_trs[y_value].append(mean_TRs)

                    synthetic_samples = add_gaussian_noise(mean_TRs, noise_level, n_synthetic_TRs)
                    for synthetic_data in synthetic_samples:
                        averaged_trs[y_value].append(synthetic_data)
                        metadata_list[y_value].append(metadata.iloc[-1, :])

                    metadata_last = metadata.iloc[-1, :]
                    metadata_list[y_value].append(metadata_last)
            else:
                raise TypeError("Loaded data is not a DataFrame. Check the file format.")

    final_data_list = []
    for y_value in range(15):
        concatenated_trs = np.concatenate(averaged_trs[y_value])
        final_data_list.append(concatenated_trs)

    final_data = pd.DataFrame(final_data_list)
    normalized_final_data = z_score_normalize(final_data, axis=0)

    """
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(normalized_final_data)

    # Convert PCA result to DataFrame
    pca_df = pd.DataFrame(pca_result)
    """

    final_metadata_list = []
    for y_value in range(15):
        averaged_metadata = pd.DataFrame(metadata_list[y_value]).mean(axis=0).to_frame().T

        if 'timepoint' in averaged_metadata.columns:
            averaged_metadata['timepoint'] = averaged_metadata['timepoint'].astype(int)
        if 'Subject' in averaged_metadata.columns:
            averaged_metadata['Subject'] = averaged_metadata['Subject'].astype(int)

        final_metadata_list.append(averaged_metadata)

    metadata_df = pd.concat(final_metadata_list, ignore_index=True)
    metadata_df['y'] = range(15)

    final_data = pd.concat([normalized_final_data, metadata_df.reset_index(drop=True)], axis=1)   # if PCA was used - use pca_df instead of normalized_final_data

    return final_data

# Function to process movie data by slicing based on the configuration
def process_movie_data(data_vis, slice, task, inputs, outputs):
    #inputs, outputs = [], []
    for movie in range(1, 15):
        movie_data = data_vis[data_vis['y'] == movie]
        input = movie_data.iloc[:, :-3]
        if slice == 'start':
            input = input.iloc[:5, :]
        elif slice == 'end':
            input = input.iloc[-1:, :]
        elif slice == 'middle':
            start_index = len(input) // 2 - 7
            input = input.iloc[start_index:start_index + 5, :]
        elif slice == 'all':
            if task == 'movies':
                all_len = 260
            else:
                all_len = 20
            zeroes = pd.DataFrame(0, index=range(all_len - len(input)), columns=input.columns)
            input = pd.concat([input, zeroes], axis=0)
        else:
            raise Exception("For now you can choose slice from [start, middle, end, all]")
        if input.empty:
            print(f"Input data for movie {movie} is empty.")
            continue
        output = [0.0 for i in range(1, 15)]
        output[movie - 1] = 1.0
        inputs.append(torch.tensor(input.values))
        outputs.append(torch.tensor(output))
    return inputs, outputs

# Function to process a directory and average voxel data based on the configuration
def process_directory(subject_folder, H, NET, NET_idx, Avg, noise_level=0.01, n_synthetic_TRs=3):
    try:
        if H == 'BOTH':
            files = glob.glob(osp.join(subject_folder, 'LH_' + NET + '*.pkl')) + \
                    glob.glob(osp.join(subject_folder, 'RH_' + NET + '*.pkl'))
            data_files = [file for file in files if not file.endswith('Avg.pkl')]
        else:
            data_files = glob.glob(osp.join(subject_folder, H + '_' + NET + '_' + str(NET_idx) + '.pkl'))
        if not data_files:
            print(f"No Default files found in {subject_folder}.")
            return

        if Avg == 1:
            averaged_data = average_voxels(data_files)
            if averaged_data is None:
                print("No valid data to average.")
                return
        elif Avg == 2:
            averaged_data = average_TRs(data_files, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)
            if averaged_data is None:
                print("No valid data to average.")
                return

        if H == 'BOTH':
            output_file = osp.join(subject_folder, H + '_' + NET + '_Avg.pkl')
        else:
            output_file = osp.join(subject_folder, H + '_' + NET + '_' + str(NET_idx) + '_Avg.pkl')
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'wb') as f:
            pickle.dump(averaged_data, f)
    except Exception as e:
        print(f"Error processing directory {subject_folder}: {e}")

# Function to create synthetic subjects with Gaussian noise
def create_synthetic_subjects(subject_folder, H, NET, NET_idx, noise_level=0.01, n_synthetic_subjects=10):
    if 'SYNTH_' in subject_folder or 'GROUP_' in subject_folder:
        return []

    synthetic_folders = []
    subject_id = osp.basename(subject_folder)

    for i in range(n_synthetic_subjects):
        current_noise_level = noise_level * (i + 1)
        synthetic_folder = f"{subject_folder}_SYNTH_{i + 1}"
        synthetic_folders.append(synthetic_folder)
        os.makedirs(synthetic_folder, exist_ok=True)

        if H == 'BOTH':
            files = glob.glob(osp.join(subject_folder, 'LH_' + NET + '*.pkl')) + \
                    glob.glob(osp.join(subject_folder, 'RH_' + NET + '*.pkl'))
        else:
            files = glob.glob(osp.join(subject_folder, H + '_' + NET + '_' + str(NET_idx) + '.pkl'))

        for file_path in files:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                if isinstance(data, pd.DataFrame):
                    voxel_data = data.iloc[:, :-3]
                    metadata = data.iloc[:, -3:]

                    synthetic_voxel_data = add_gaussian_noise(voxel_data.to_numpy(), current_noise_level, n_synthetic=1)[0]
                    synthetic_data = pd.concat([pd.DataFrame(synthetic_voxel_data, columns=voxel_data.columns), metadata], axis=1)

                    synthetic_file_path = osp.join(synthetic_folder, osp.basename(file_path))
                    with open(synthetic_file_path, 'wb') as syn_file:
                        pickle.dump(synthetic_data, syn_file)
                else:
                    raise TypeError("Loaded data is not a DataFrame. Check the file format.")

    return synthetic_folders

# Function to shuffle labels
def shuffle_labels(labels):
    np.random.shuffle(labels)
    return labels

# Function to shuffle a DataFrame
def shuffle_df(df):
    matrix = df.to_numpy()
    np.random.shuffle(matrix)
    matrix = matrix.T
    np.random.shuffle(matrix)
    matrix = matrix.T
    return pd.DataFrame(matrix, columns=df.columns)

# Function to get dataloaders for training, evaluation, and testing (called on ly in normal training, not in cross validation)
def get_dataloaders2(directory, NET, NET_idx, H, slice, batch_size, task, Avg=0, noise_level=0.01,
                     n_synthetic_TRs=3, Create_synthetic_subjects=0, n_synthetic_subjects=10, Group_by_subjects=0, group_size=10, use_original_for_val_test=False, fold=None):
    dataloaders = {}
    file_exists = True

    for phase in ['train', 'eval', 'test']:
        inputs = []  # Initialize inputs here
        outputs = []  # Initialize outputs here

        phase_path = osp.join(directory, phase, task)
        subject_folders = list(glob.iglob(phase_path + '/**'))

        for folder in subject_folders:
            if 'SYNTH_' in folder or 'GROUP_' in folder:
                print(f"Removing synthetic subject folder: {folder}")
                for file in glob.glob(osp.join(folder, '*')):
                    os.remove(file)
                os.rmdir(folder)
        subject_folders = [folder for folder in subject_folders if 'SYNTH_' not in folder and 'GROUP_' not in folder]

        if Create_synthetic_subjects == 1 and (use_original_for_val_test == False or (use_original_for_val_test == True and phase == 'train')):
            for subject_folder in subject_folders:
                synthetic_folders = create_synthetic_subjects(subject_folder, H, NET, NET_idx, noise_level=noise_level, n_synthetic_subjects=n_synthetic_subjects)
                subject_folders.extend(synthetic_folders)

        if Group_by_subjects == 1:
            for folder in subject_folders:
                if 'GROUP_' in folder:
                    print(f"Removing subjects group folder: {folder}")
                    for file in glob.glob(osp.join(folder, '*')):
                        os.remove(file)
                    os.rmdir(folder)
            subject_folders = [folder for folder in subject_folders if 'GROUP_' not in folder]

            random.shuffle(subject_folders)

            group_folders_list = []
            for i in range(0, len(subject_folders), group_size):
                if len(subject_folders[i:i + group_size]) < group_size:
                    break
                group_folders = subject_folders[i:i + group_size]
                group_folder_name = f"GROUP_{osp.basename(group_folders[0])}_{osp.basename(group_folders[-1])}"
                if fold is not None:
                    group_folder_name += f"_fold_{fold}"
                group_folder_path = osp.join(phase_path, group_folder_name)
                os.makedirs(group_folder_path, exist_ok=True)

                for H_ in ['LH', 'RH']:
                    if H_ == H or H == 'BOTH':
                        for net_file in glob.glob(osp.join(group_folders[0], H_ + '_' + NET + '*.pkl')):
                            net_files = [osp.join(folder, osp.basename(net_file)) for folder in group_folders]
                            averaged_data = average_subjects(net_files)
                            output_file = osp.join(group_folder_path, osp.basename(net_file))
                            with open(output_file, 'wb') as f:
                                pickle.dump(averaged_data, f)
                group_folders_list.append(group_folder_path)
            subject_folders = group_folders_list

            for group_folder in group_folders_list:
                process_directory(group_folder, H, NET, NET_idx, Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)

        for subject_folder in subject_folders:
            if Avg in [1, 2]:
                process_directory(subject_folder, H, NET, NET_idx, Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)

            try:
                if Avg in [1, 2]:
                    if H == "BOTH":
                        file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, 'Avg'))
                    else:
                        file_path = osp.join(subject_folder, '{}_{}_{}_{}.pkl'.format(H, NET, NET_idx, 'Avg'))
                else:
                    file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, NET_idx))
                if not osp.exists(file_path):
                    print(f"File does not exist: {file_path}")
                    continue

                with open(file_path, 'rb') as file:
                    data_vis = pickle.load(file)
                num_voxels = data_vis.shape[1] - 3

                inputs, outputs = process_movie_data(data_vis, slice, task, inputs, outputs)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
            except FileNotFoundError:
                file_exists = False
                continue

        if not file_exists:
            continue

        tensor_inputs = torch.stack(inputs)
        tensor_inputs = tensor_inputs.double()
        labels = torch.stack(outputs)

        if phase == 'train':
            train_labels = labels
            train_dataset = TensorDataset(tensor_inputs, train_labels)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            dataloaders[phase] = train_dataloader
        elif phase == 'eval':
            eval_labels = labels
            eval_dataset = TensorDataset(tensor_inputs, eval_labels)
            eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
            dataloaders['val'] = eval_dataloader
        else:
            # test_labels = shuffle_labels(labels) # Tali - 22_7
            test_labels = labels
            test_dataset = TensorDataset(tensor_inputs, test_labels)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            dataloaders[phase] = test_dataloader

    if not file_exists:
        return None, None, None, None

    return dataloaders['train'], dataloaders['val'], dataloaders['test'], num_voxels

# Function to average subject data after grouping several subjects together
def average_subjects(data_files):
    averaged_data_list = []
    metadata_list = []

    for file_path in data_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, pd.DataFrame):
                voxel_data = data.iloc[:, :-3]
                metadata = data.iloc[:, -3:]
                averaged_data_list.append(voxel_data)
                metadata_list.append(metadata)
            else:
                raise TypeError("Loaded data is not a DataFrame. Check the file format.")

    if not averaged_data_list:
        return None

    concatenated_data = pd.concat(averaged_data_list).groupby(level=0).mean()
    concatenated_metadata = pd.concat(metadata_list).groupby(level=0).mean()

    normalized_data = z_score_normalize(concatenated_data, axis=0)

    final_data = pd.concat([normalized_data, concatenated_metadata], axis=1)

    return final_data

# Helper function for cross-validation dataloaders
def get_dataloaders2_helper(data_files, NET, NET_idx, H, slice, batch_size, task, Avg, noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix):
    inputs, outputs = [], []
    subject_folders = sorted(list(set([osp.dirname(file) for file in data_files])))

    # In cross_val we run with grouping by subjects so we don't want to create synhetic subjects and then group them - this code is removed
    #if Create_synthetic_subjects == 1:
    #    for subject_folder in subject_folders:
    #        synthetic_folders = create_synthetic_subjects(subject_folder, H, NET, NET_idx, noise_level=noise_level, n_synthetic=10)
    #        subject_folders.extend(synthetic_folders)

    if Group_by_subjects == 1:
        #random.shuffle(subject_folders)
        group_folders_list = []
        for i in range(0, len(subject_folders), group_size):
            if len(subject_folders[i:i + group_size]) < group_size:
                break
            group_folders = subject_folders[i:i + group_size]
            group_folder_name = f"GROUP_{osp.basename(group_folders[0])}_{osp.basename(group_folders[-1])}_{run_suffix}"
            group_folder_path = osp.join(osp.dirname(group_folders[0]), group_folder_name)
            os.makedirs(group_folder_path, exist_ok=True)

            for H_ in ['LH', 'RH']:
                if H_ == H or H == 'BOTH':
                    for net_file in glob.glob(osp.join(group_folders[0], H_ + '_' + NET + '*.pkl')):
                        net_files = [osp.join(folder, osp.basename(net_file)) for folder in group_folders]
                        averaged_data = average_subjects(net_files)
                        output_file = osp.join(group_folder_path, osp.basename(net_file))
                        with open(output_file, 'wb') as f:
                            pickle.dump(averaged_data, f)
            group_folders_list.append(group_folder_path)
        subject_folders = group_folders_list

        for group_folder in group_folders_list:
            process_directory(group_folder, H, NET, NET_idx, Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)

    for subject_folder in subject_folders:
        if Avg in [1, 2]:
            process_directory(subject_folder, H, NET, NET_idx, Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)

        try:
            if Avg in [1, 2]:
                if H == "BOTH":
                    file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, 'Avg'))
                else:
                    file_path = osp.join(subject_folder, '{}_{}_{}_{}.pkl'.format(H, NET, NET_idx, 'Avg'))
            else:
                file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, NET_idx))
            if not osp.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue

            with open(file_path, 'rb') as file:
                data_vis = pickle.load(file)
            num_voxels = data_vis.shape[1] - 3
            inputs, outputs = process_movie_data(data_vis, slice, task, inputs, outputs)


            #print(f"Processed {file_path}, inputs length: {len(inputs)}, outputs length: {len(outputs)}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
        except FileNotFoundError:
            continue

    if len(inputs) == 0 or len(outputs) == 0:
        return None, None

    tensor_inputs = torch.stack(inputs)
    tensor_inputs = tensor_inputs.double()
    labels = torch.stack(outputs)

    dataset = TensorDataset(tensor_inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, num_voxels

# Function to get dataloaders for cross-validation
def get_dataloaders2_for_cross_val(data_files, NET, NET_idx, H, batch_size, slice, task, Avg, noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix):
    files = [file for i, file in enumerate(data_files)]
    dataloader, num_voxels = get_dataloaders2_helper(files, NET, NET_idx, H, slice, batch_size, task, Avg, noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix)
    return dataloader, num_voxels