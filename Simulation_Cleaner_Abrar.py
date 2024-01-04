import os
import shutil

# Path to your original dataset folder
original_folder = r'F:\CvDataset\Dataset\batch_20230912_part2\Part2'

# Create a new folder for corrected simulations
new_folder = r'F:\CvDataset\Dataset\batch_20230912_part2\Part2_Cleaned'
os.makedirs(new_folder, exist_ok=True)

total_simulations = 0
valid_simulations = 0
deleted_simulations = 0

for i in range(5500, 11000):  # Assuming the range goes from 0 to 5499
    simulation_prefix = f"0_{i}"

    total_simulations += 1

    # Check if all 13 files exist for the current simulation
    files_exist = all((
        os.path.exists(os.path.join(original_folder, f"{simulation_prefix}_GT_pose_0_thermal.png")),
        os.path.exists(os.path.join(original_folder, f"{simulation_prefix}_Parameters.txt")),
        *[os.path.exists(os.path.join(original_folder, f"{simulation_prefix}_pose_{j}_thermal.png")) for j in range(11)]
    ))

    if files_exist:
        valid_simulations += 1
        print(f"Copying simulation {simulation_prefix}")
        for file_type in ["GT_pose_0_thermal.png", "Parameters.txt", *[f"pose_{j}_thermal.png" for j in range(11)]]:
            src_path = os.path.join(original_folder, f"{simulation_prefix}_{file_type}")
            dst_path = os.path.join(new_folder, f"{simulation_prefix}_{file_type}")
            shutil.copy2(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
    else:
        deleted_simulations += 1
        print(f"Deleting simulation {simulation_prefix}")
        for file_type in ["GT_pose_0_thermal.png", "Parameters.txt", *[f"pose_{j}_thermal.png" for j in range(11)]]:
            file_path = os.path.join(original_folder, f"{simulation_prefix}_{file_type}")
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted {file_path}")

print("Cleaning process complete for batch_20230912_part1")
print(f"Total Simulations: {total_simulations}")
print(f"Valid Simulations: {valid_simulations}")
print(f"Deleted Simulations: {deleted_simulations}")
