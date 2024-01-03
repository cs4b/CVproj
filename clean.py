import os
import glob
import shutil
import re

base_dirs = ['Part1_1', 'Part1_2', 'Part2_1', 'Part2_2', 'Part3_1', 'Part3_2']

pattern = r"\d_\d+_pose_\d+_thermal.png"
gt_pattern = r"\d_\d+_GT_pose_\d+_thermal.png"
param_pattern = r"\d_\d+_Parameters.txt"


output_dir = 'complete_sets'
os.makedirs(output_dir, exist_ok=True)


with open('complete_sets.txt', 'w') as complete_set_file, \
        open('incomplete_sets.txt', 'w') as incomplete_set_file, \
        open('summary.txt', 'w') as summary_file:
    for base_dir in base_dirs:
        files = glob.glob(os.path.join(base_dir, "*")) #these are all the files in the directory

        groups = {} # create groups based on the patterns
        for file in files:
            base = os.path.basename(file)
            if re.match(pattern, base) or re.match(gt_pattern, base) or re.match(param_pattern, base):
                parts = base.split('_')
                key = tuple(parts[:2])
                if key not in groups:
                    groups[key] = []
                groups[key].append(file)

        #if the set is complete, then write it to a subdirectory and logit, if its not, then just log it
        complete_sets = []
        incomplete_sets = []
        for group in groups.values():
            if len(group) == 13:
                for file in group:
                    complete_set_file.write(file + '\n')
                #This below is the copy (2 lines), so uncomment it, when you are sure everything is right..
                #otherwise it copies 45GBs every runtime.
                #for file in group:
                #    shutil.copy(file, os.path.join(output_dir, os.path.basename(file)))
                complete_sets.append(group)
            else:
                #incomplete log
                for key in group:
                    incomplete_set_file.write('_'.join(map(str, key)) + '\n')
                incomplete_sets.append(group)

        # log number of complete and incomplete sets and total number of files to the summary file
        summary_file.write(
            f'Number of complete sets: {len(complete_sets)}, Total number of files in complete sets: {sum(len(group) for group in complete_sets)}\n')
        summary_file.write(
            f'Number of incomplete sets: {len(incomplete_sets)}, Total number of files in incomplete sets: {sum(len(group) for group in incomplete_sets)}\n')