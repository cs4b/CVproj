from keras.models import load_model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

#This is a helper function to read images:
def read_and_normalize_images(folder_path):
    normalized_images = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            normalized_img = img.astype('float32') / 255.0
            normalized_images.append(normalized_img)

    return np.array(normalized_images)

#This is a helper function to visualise predictions without groundtruth
def visualize_predictions_outside(model, images):
    predictions = model.predict(images)
    num_images = (len(images))
    #fig, axes = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))
    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 5))

    #range specifies how many integral images we visualise from the real_integrals directory
    for i in range(len(images)):
        axes[0, i].imshow(images[i], cmap='gray')
        axes[0, i].set_title('Original Image')
        axes[0, i].axis('off')

        axes[1, i].imshow(predictions[i, :, :, 0], cmap='gray')
        axes[1, i].set_title('Predicted Image')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_predictions_and_save(model, images, save_folder):

    # Ensure the save folder exists
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    predictions = model.predict(images)
    num_images = len(images)

    for i in range(num_images):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(images[i], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(predictions[i, :, :, 0], cmap='gray')
        axes[1].set_title('Predicted Image')
        axes[1].axis('off')

        plt.tight_layout()

        plt.savefig(os.path.join(save_folder, f'result_{i+1}.png'))

        plt.close()

#Loading the model
model_path = r"D:\proj\model_separated"
loaded_model = load_model(model_path)

#Path to test_set
test_path = r"D:\proj\to_train\testfolder\testset"

images = read_and_normalize_images(folder_path=test_path)
visualize_predictions_outside(loaded_model,images)

#Save the predictions to a folder
save_path = r"D:\proj\to_train\testfolder\testset\test_pred"
if not os.path.exists(save_path):
        os.makedirs(save_path)

visualize_predictions_and_save(loaded_model,images,save_path)