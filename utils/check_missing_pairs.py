import os

# Function to check if both .mp4 and .jsonl files exist for a given filename
def has_pair(filename):
    mp4_file = folder_path + filename + ".mp4"
    jsonl_file = folder_path + filename + ".jsonl"
    return os.path.isfile(mp4_file) and os.path.isfile(jsonl_file)

# Replace 'folder_path' with the actual path to your folder containing the files
folder_path = '/mnt/data/plc2000/2500_MineRLBasaltFindCave-v0/'

# List all files in the folder
files = os.listdir(folder_path)

# Remove the file extension to get the filename (without '.mp4' or '.jsonl')
file_names = set(os.path.splitext(file)[0] for file in files)

# Find the filenames missing a pair and print them
for filename in file_names:
    if not has_pair(filename):
        os.remove(folder_path + filename + ".mp4")
        print(filename)
