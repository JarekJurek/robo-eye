import os

directory_path = "robo-eye/detection/predictions/labels_filtered"

for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory_path, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        filtered_lines = [line for line in lines if line.split()[0] in {"0", "1", "2"}]

        with open(file_path, "w") as file:
            file.writelines(filtered_lines)

print("Processing complete.")
