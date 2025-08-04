import os


def remove_existing_files(txt_path, image_folder, new_txt_path):
    # Read raw txt file
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # Check each file for existence
    non_existing_files = []
    for line in lines:
        file_name = line.strip()
        file_path = os.path.join(image_folder, file_name)
        if not os.path.exists(file_path):
            non_existing_files.append(line)
        else:
            print(f"File exists and will be removed: {file_name}")

    # Write non-existent filenames to a new text file
    with open(new_txt_path, 'w') as f:
        f.writelines(non_existing_files)


if __name__ == "__main__":
    original_txt_path = 'e-ophtha_MA_trainset.txt'
    image_folder_path = 'e-ophtha_MA/mask'
    new_txt_path = 'e-ophtha_MA_trainset4AD.txt'
    remove_existing_files(original_txt_path, image_folder_path, new_txt_path)
