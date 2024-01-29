import os

def get_all_file_paths(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


paths = get_all_file_paths('QA/Acc+')

for path in paths:
    print(path)
