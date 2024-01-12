import glob
import os
import shutil
from PIL import Image
from tqdm import tqdm


def separate_images(source_dir, destination_dir):
    """
    Separates images into iterations and GT
    :param source_dir: Directory containing images to separate
    :param destination_dir: Directory to put the separated images
    :return: None
    """
    # check if source directory exists
    if not os.path.isdir(source_dir):
        raise ValueError(f'Source directory {source_dir} does not exist')

    # Create the destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Create the GT folder if it doesn't exist
    if not os.path.exists(os.path.join(destination_dir, 'GT')):
        os.makedirs(os.path.join(destination_dir, 'GT'))

    # get all images from the source directory
    images = sorted(glob.glob(os.path.join(source_dir, "*.png")))

    # Copy GT images to GT folder
    for image in tqdm(images, desc='separating images'):
        if 'GT' in os.path.basename(image):
            shutil.copy(os.path.join(image), os.path.join(destination_dir, 'GT'))
        else:
            unique_folder = os.path.basename(image).split('_')[0:2][0] + '_' + os.path.basename(image).split('_')[0:2][
                1]
            if not os.path.exists(os.path.join(destination_dir, unique_folder)):
                os.makedirs(os.path.join(destination_dir, unique_folder))
            shutil.copy(os.path.join(image), os.path.join(destination_dir, unique_folder))


separate_images(r'\proj\complete_sets', r'\proj\separated')