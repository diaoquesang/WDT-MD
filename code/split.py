import random
import os


def traverse_directory(directory, txt_name):
    # Create an empty list for storing filenames
    file_names = []
    # Iterate over all files and subdirectories in a directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
    # Sort by file name
    file_names.sort()
    # Create a new txt file and write the filename to it
    with open(txt_name, 'w') as f:
        for file_name in file_names:
            f.write(file_name + '\n')


def split_dataset(file_path, train_ratio=0.8, val_ratio=0.1, former=None):
    # Read the dataset
    with open(file_path, 'r') as f:
        data = f.readlines()
    # Randomly shuffle datasets
    random.shuffle(data)

    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)

    train_set = data[:train_size]
    val_set = data[train_size:train_size + val_size]
    test_set = data[train_size + val_size:]
    with open(former + '_trainset.txt', 'w') as f:
        f.writelines(train_set)
    with open(former + '_valset.txt', 'w') as f:
        f.writelines(val_set)
    with open(former + '_testset.txt', 'w') as f:
        f.writelines(test_set)
    print(f"Finished.")


if __name__ == "__main__":
    traverse_directory('e-ophtha_MA/image', "e-ophtha_MA.txt")
    split_dataset('e-ophtha_MA.txt', 0.8, 0.1, "e-ophtha_MA")
