import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import matplotlib.pyplot as plt

def read_and_normalize_images(folder_path):
    normalized_images = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            normalized_img = img.astype('float32') / 255.0
            normalized_images.append(normalized_img)

    return np.array(normalized_images)

def visualize_predictions_outside(model, images):
    predictions = model.predict(images)
    num_images = min(len(images), 2)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))

    for i in range(num_images):
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(predictions[i, :, :, 0], cmap='gray')
        axes[i, 1].set_title('Predicted Image')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    # Load the trained U-Net model
    saved_model_path = r'F:\\CvDataset\\Dataset\\model'
    loaded_model = load_model(saved_model_path)

    # Test images folder path
    test_images_path = r'F:\CvDataset\Dataset\Seperated_Folders\0_2834'
    test_images = read_and_normalize_images(test_images_path)

    # Visualize predictions
    visualize_predictions_outside(loaded_model, test_images)

if __name__ == "__main__":
    main()
