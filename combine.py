import os
import shutil

def copy_images_from_subdirectories(src_folder_postfix, src_folder_files, dest_folder):
    # Iterate over subdirectories in the source folder for postfix extraction
    for subfolder_name in os.listdir(src_folder_postfix):
        subfolder_path = os.path.join(src_folder_postfix, subfolder_name)

        # Check if the path is a directory
        if os.path.isdir(subfolder_path):
            # Extract the last two parts of the subfolder name as postfix
            postfix = '_'.join(subfolder_name.split('_')[-2:])

            # Construct the source and destination paths for the image file
            image_filename = f"{postfix}_GT_pose_0_thermal.png"
            src_image_path = os.path.join(src_folder_files, image_filename)
            dest_image_path = os.path.join(dest_folder, image_filename)

            # Check if the file exists in the source files folder
            if os.path.exists(src_image_path):
                # Copy the file to the destination folder
                shutil.copy2(src_image_path, dest_image_path)
                print(f"Copied: {image_filename} from {subfolder_name}")
            else:
                print(f"File not found in {src_folder_files}: {image_filename}")

# Example usage:
src_folder_postfix = "D:\proj\separated\integraltest\integrals"  # Folder for postfix extraction
src_folder_files = "D:\proj\separated\GT"      # Folder containing files to match
dest_folder = r"D:\proj\to_train"

copy_images_from_subdirectories(src_folder_postfix, src_folder_files, dest_folder)
