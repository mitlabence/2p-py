from typing import List

import labrotation.file_handling as fh
import hashlib
import os


def sha256_of_file(fpath: str) -> str:
    """
    Given the absolute path to a file,return the SHA256 hash
    :param fpath: absolute path with filename
    :return: the sha256 hexadecimal hash value as string
    """
    with open(fpath, "rb") as f:
        file_as_bytes = f.read()  # read entire file as bytes
        hash_val = hashlib.sha256(file_as_bytes).hexdigest()
        return hash_val


def sha256_of_all_in_dir(directory: str) -> List[str]:
    """
    Given the directory path, calculate the SHA256 hash for all files (recursively in subdirectories as well).
    :param directory: the directory path in which to loop through all files and files in subfolders recursively.
    :return: a list of tuples, two str entries each: (absolute file path, hash value)
    """
    file_hash_pairs = []  # list of tuples: (file path, hash), type: (str, str)
    for root, dirs, files in os.walk(directory):
        for filename in files:
            fpath = os.path.normpath(os.path.join(root, filename))
            file_hash_pairs.append((fpath, sha256_of_file(fpath)))
    return file_hash_pairs


def sha256_dir_to_txt(directory: str, output_fpath: str) -> None:
    """
    Given directory path and output file path (including file name), save the results of sha256 hash of all files and
    content of subfolders to a txt file determined in output_fpath
    :param directory: the directory whose members will be the target of sha256 hash calculations.
    :param output_fpath: the file path and file name of the resulting txt file (".txt" extension at the end optional)
    :return: None
    """
    if output_fpath[-4:] == ".txt":
        output_file_path = output_fpath
    else:
        output_file_path = output_fpath + ".txt"
    file_hash_pairs = sha256_of_all_in_dir(directory)
    with open(output_file_path, "w") as output_file:
        for file_hash_tuple in file_hash_pairs:
            output_file.write(file_hash_tuple[0] + " : " + file_hash_tuple[1] + "\n")
    print(f"sha256_dir_to_txt: Successfully written results to\n{output_file_path}")


filename = fh.get_filename_with_date("test_output")
output_filepath = fh.choose_dir_for_saving_file("asd", filename)
input_directory = fh.open_dir("Choose directory to run SHA256 on")
sha256_dir_to_txt(input_directory, output_filepath)
