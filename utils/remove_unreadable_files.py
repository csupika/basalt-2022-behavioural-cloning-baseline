import os

def remove_unreadable_files(data):
    # Read the contents of data
    with open(data, 'r') as file:
        lines = file.readlines()

    # Extract file names using a list comprehension
    extracted_corrupted_files = [line.split()[-1] for line in lines if "Could not read frame from video" in line]
    extracted_missing_files = [line.split()[-1][1:-1] for line in lines if "FileNotFoundError:" in line]

    # Remove duplicates using set comprehension (instead of converting list to set and back to list)
    extracted_corrupted_files = list(set(extracted_corrupted_files))
    extracted_missing_files = list(set(extracted_missing_files))

    # Generate jsonl pairs of files that couldn't be read using a list comprehension
    corrupted_files_could_not_be_read = [file for file in extracted_corrupted_files for _ in range(2)]
    missing_files_could_not_be_read = [file for file in extracted_missing_files for _ in range(2)]

    # Replace the extension in the second element of each pair
    corrupted_files_could_not_be_read[1::2] = [file.replace('mp4', 'jsonl') for file in corrupted_files_could_not_be_read[1::2]]
    missing_files_could_not_be_read[0::2] = [file.replace('jsonl', 'mp4') for file in missing_files_could_not_be_read[1::2]]

    # Write the corrupted_files_could_not_be_read list to a new file
    with open("removed_files.txt", 'a+') as output_file:
        for file_path in corrupted_files_could_not_be_read:
            output_file.write(file_path + '\n')
        for file_path in missing_files_could_not_be_read:
            output_file.write(file_path + '\n')

    # Remove the files in resulting list
    for file_path in corrupted_files_could_not_be_read:
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except OSError as e:
            print(f"Error while trying to remove file {file_path}: {e}")

    for file_path in missing_files_could_not_be_read:
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except OSError as e:
            print(f"Error while trying to remove file {file_path}: {e}")

if __name__ == '__main__':
    log_list = os.listdir("../logs/train")
    for log in log_list:
        print(log)
        remove_unreadable_files(data= "../logs/train/" + log)
