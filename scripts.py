import json
import csv
import numpy as np


def store_adjacency_matrix(matrix, filename):
    np.savetxt(filename, matrix, delimiter=",")


def store_matrix_parameters(data, filename):
    # Store Data in Json Format
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def write_all_epochs_to_csv(file_name, pth_file, sorted_values):
    # Open the CSV file and write the data
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([pth_file] + sorted_values)


def excel_column_letter_to_number(column_letter):
    column_letter = column_letter.upper()  # Ensure the column letter is in uppercase
    column_number = 0
    length = len(column_letter)

    for i, letter in enumerate(column_letter):
        column_number += (ord(letter) - ord('A') + 1) * (26 ** (length - i - 1))

    return column_number