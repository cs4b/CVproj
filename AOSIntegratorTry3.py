import numpy as np
import cv2
import os
import math
from LFR_utils import read_poses_and_images, pose_to_virtualcamera, init_aos, init_window
import LFR_utils as utils
import pyaos
import glm
import re
import numpy as np
import glob

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def integrate(directory_path, output_base_path):
    Download_Location = output_base_path
    print(Download_Location)
    Integral_Path = os.path.join(Download_Location, 'Integrals')
    if not os.path.exists(Integral_Path):
        os.mkdir(Integral_Path)
    else:
        print(f"The directory '{Integral_Path}' already exists.")

    #############################Start the AOS Renderer###############################################################
    w, h, fovDegrees = 512, 512, 50
    render_fov = 50

    if 'window' not in locals() or window == None:
        window = pyaos.PyGlfwWindow(w, h, 'AOS')

    aos = pyaos.PyAOS(w, h, fovDegrees)
    set_folder = r'C:\MyPythonProjects\JKU\Sem1\CV\AOS-stable_release\AOS-stable_release\AOS for Drone Swarms\LFR\python'
    aos.loadDEM(os.path.join(set_folder, 'zero_plane.obj'))

    #############################Create Poses for Initial Positions###############################################################

    def eul2rotm(theta):
        s_1 = math.sin(theta[0])
        c_1 = math.cos(theta[0])
        s_2 = math.sin(theta[1])
        c_2 = math.cos(theta[1])
        s_3 = math.sin(theta[2])
        c_3 = math.cos(theta[2])
        rotm = np.identity(3)
        rotm[0, 0] = c_1 * c_2
        rotm[0, 1] = c_1 * s_2 * s_3 - s_1 * c_3
        rotm[0, 2] = c_1 * s_2 * c_3 + s_1 * s_3

        rotm[1, 0] = s_1 * c_2
        rotm[1, 1] = s_1 * s_2 * s_3 + c_1 * c_3
        rotm[1, 2] = s_1 * s_2 * c_3 - c_1 * s_3

        rotm[2, 0] = -s_2
        rotm[2, 1] = c_2 * s_3
        rotm[2, 2] = c_2 * c_3

        return rotm

    def createviewmateuler(eulerang, camLocation):
        rotationmat = eul2rotm(eulerang)
        translVec = np.reshape((-camLocation @ rotationmat), (3, 1))
        conjoinedmat = (np.append(np.transpose(rotationmat), translVec, axis=1))
        return conjoinedmat

    def divide_by_alpha(rimg2):
        a = np.stack((rimg2[:, :, 3], rimg2[:, :, 3], rimg2[:, :, 3]), axis=-1)
        return rimg2[:, :, :3] / a

    def pose_to_virtualcamera(vpose):
        vp = glm.mat4(*np.array(vpose).transpose().flatten())
        ivp = glm.inverse(glm.transpose(vp))
        Posvec = glm.vec3(ivp[3])
        Upvec = glm.vec3(ivp[1])
        FrontVec = glm.vec3(ivp[2])
        lookAt = glm.lookAt(Posvec, Posvec + FrontVec, Upvec)
        cameraviewarr = np.asarray(lookAt)
        return cameraviewarr

    Numberofimages = 11
    Focal_plane_values = [0, -0.75, -1.5, -2.25, -3]
    ref_loc = [[5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    altitude_list = [35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35]
    center_index = 5

    site_poses = []
    for i in range(Numberofimages):
        EastCentered = (ref_loc[0][i] - 0.0)
        NorthCentered = (0.0 - ref_loc[1][i])
        M = createviewmateuler(np.array([0.0, 0.0, 0.0]), np.array([ref_loc[0][i], ref_loc[1][i], -altitude_list[i]]))
        ViewMatrix = np.vstack((M, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)))
        camerapose = np.asarray(ViewMatrix.transpose(), dtype=np.float32)
        site_poses.append(camerapose)

    #############################Read the generated images from the simulator and store in a list ###############################################################

    numbers = re.compile(r'(\d+)')
    imagelist = []
    for img in sorted(glob.glob(directory_path + '/*.png'), key=numericalSort):
        n = cv2.imread(img)
        imagelist.append(n)

    # Create simulation-specific folder
    simulation_folder = os.path.join(Integral_Path, f'integrals_{os.path.basename(directory_path)}')
    if not os.path.exists(simulation_folder):
        os.mkdir(simulation_folder)
    else:
        print(f"The directory '{simulation_folder}' already exists.")

    for focal_plane in Focal_plane_values:
        print(f"Rendering images for focal length: {focal_plane}")
        aos.clearViews()
        aos.setDEMTransform([0, 0, focal_plane])

        for i in range(len(imagelist)):
            print(f"Rendering image {i + 1}/{len(imagelist)}")
            # Print relevant information about the image being processed
            print(f"Image path: {image_files[i]}")
            aos.addView(imagelist[i], site_poses[i], "DEM BlobTrack")

        proj_RGBimg = aos.render(pose_to_virtualcamera(site_poses[center_index]), render_fov)
        tmp_RGB = divide_by_alpha(proj_RGBimg)

        filename = f'integral_{focal_plane}.png'
        output_file_path = os.path.join(simulation_folder, filename)
        cv2.imwrite(output_file_path, tmp_RGB)

        # Print image properties for debugging
        print(f"Image: {output_file_path}")
        print(f"  Min value: {np.min(tmp_RGB)}")
        print(f"  Max value: {np.max(tmp_RGB)}")
        print(f"  Mean value: {np.mean(tmp_RGB)}")
        print(f"  Std value: {np.std(tmp_RGB)}")

if __name__ == "__main__":
    base_path = r'F:\CvDataset\Dataset\smpl'
    output_path = r'F:\CvDataset\Dataset\Sample_Integrals'

    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)

        image_files = sorted(glob.glob(os.path.join(subdir_path, '*.png')), key=numericalSort)
        if len(image_files) == 11:
            integrate(subdir_path, output_path)
