import re
import numpy as np

# Read the file content
file_path = 'PostPruning-50.txt'
with open(file_path, 'r') as file:
    content = file.read()

# Regular expressions to match the required metrics
total_training_time_pattern = re.compile(r"Total Training Time:\s*([\d.]+)\s*seconds")
entire_training_energy_pattern = re.compile(r"Entire Training Energy:\s*([\d.]+)\s*J")
average_inference_time_pattern = re.compile(r"Average Inference Time per Batch:\s*([\d.]+)\s*seconds")

# Find all matches in the file
total_training_times = [float(x) for x in total_training_time_pattern.findall(content)]
entire_training_energies = [float(x) for x in entire_training_energy_pattern.findall(content)]
average_inference_times = [float(x) for x in average_inference_time_pattern.findall(content)]

# Calculate the averages
average_training_time = np.mean(total_training_times)
average_training_energy = np.mean(entire_training_energies)
average_inference_time = np.mean(average_inference_times)

print(f"Average Total Training Time: {average_training_time:.2f} seconds")
print(f"Average Entire Training Energy: {average_training_energy:.2f} J")
print(f"Average Inference Time per Batch: {average_inference_time:.6f} seconds")