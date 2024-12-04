import os

input_directory = "robo-eye/detection/predictions/labels_filtered"
output_directory = "robo-eye/detection/predictions/labels_filtered"

os.makedirs(output_directory, exist_ok=True)

image_width = 1224
image_height = 370

for filename in os.listdir(input_directory):
    if filename.endswith(".txt"):
        input_file_path = os.path.join(input_directory, filename)
        output_file_path = os.path.join(output_directory, filename)

        with open(input_file_path, "r") as file:
            lines = file.readlines()

        processed_lines = []
        for line in lines:
            parts = line.split()
            if parts[0] in {"0", "1", "2"}:  # Keep only the required classes
                class_id = parts[0]
                probability = parts[1]
                center_x = float(parts[2]) * image_width
                center_y = float(parts[3]) * image_height
                bbox_width = float(parts[4]) * image_width
                bbox_height = float(parts[5]) * image_height
                processed_line = f"{class_id} {probability} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
                processed_lines.append(processed_line)

        with open(output_file_path, "w") as file:
            file.writelines(processed_lines)

print("Processing complete.")
