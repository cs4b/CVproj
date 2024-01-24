import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models, Input
from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)]
            )
    except RuntimeError as e:
        print(e)
#tf.config.experimental.set_visible_devices([], 'GPU')
image_dir = 'D:\proj\separated\integraltest\integrals'
label_dir = 'D:\proj\separated\GT'
image_paths = []
label_paths = []

#this is the imgloader while im reading all integrals
"""for dir in os.listdir(image_dir):
  image_subdir = os.path.join(image_dir, dir)
  for image_path in os.listdir(image_subdir):
      image_paths.append(os.path.join(image_subdir, image_path))
      id = dir.replace("integrals_", "")
      label_name = f"{id}_GT_pose_0_thermal.png"
      label_paths.append(os.path.join(label_dir, label_name))
"""
#this is the imgloader while im reading 1 integral
for dir in os.listdir(image_dir):
    image_subdir = os.path.join(image_dir, dir)
    if os.path.isdir(image_subdir):  # Check if it's a directory
        # Take only the first image file in the directory
        image_files = [f for f in os.listdir(image_subdir) if os.path.isfile(os.path.join(image_subdir, f))]
        if image_files:  # Check if there are any files in the directory
            image_path = os.path.join(image_subdir, image_files[0])
            image_paths.append(image_path)

            id = dir.replace("integrals_", "")
            label_name = f"{id}_GT_pose_0_thermal.png"
            label_path = os.path.join(label_dir, label_name)
            label_paths.append(label_path)

def load_samples(image_paths, label_paths, num_samples):
    images = []
    labels = []

    for image_path, label_path in zip(image_paths[:num_samples], label_paths[:num_samples]):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Load label as grayscale

        if image is None or label is None:
            print(f"Warning: Could not read image or label file at {image_path} or {label_path}. Skipping this sample.")
            continue
        # Resize images and labels to a consistent size
        image = cv2.resize(image, (512, 512))
        label = cv2.resize(label, (512, 512))

        images.append(image)
        labels.append(label)

    images_normalized = [image / 255.0 for image in images]
    labels_normalized = [label / 255.0 for label in labels]
    threshold = 0.88
    binary_mask = tf.where(np.array(labels_normalized) > threshold,1,0)

    return np.array(images_normalized), np.array(binary_mask)

#This is needed for the testing
def read_and_normalize_images(folder_path):
    normalized_images = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            normalized_img = img.astype('float32') / 255.0
            normalized_images.append(normalized_img)

    return np.array(normalized_images)

def unet_model(input_shape):
    inputs = Input(shape=input_shape)

    # Contracting
    conv1 = layers.Conv2D(32, 3, activation=None, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(32, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Middle
    conv3 = layers.Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)

    # Expansive
    up4 = layers.UpSampling2D(size=(2, 2))(conv3)
    concat4 = layers.concatenate([conv2, up4], axis=-1)
    conv4 = layers.Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(concat4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)

    up5 = layers.UpSampling2D(size=(2, 2))(conv4)
    concat5 = layers.concatenate([conv1, up5], axis=-1)
    conv5 = layers.Conv2D(32, 3, activation=None, padding='same', kernel_initializer='he_normal')(concat5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(32, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)

    # Output layer
    output = layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = models.Model(inputs=inputs, outputs=output)

    return model

def visualize_predictions(model, xv, yv):
    """
    Visualizes the input image, ground truth, and model predictions.

    Parameters:
    - model: The trained U-Net model.
    - input_image: The input image for visualization.
    - ground_truth: The ground truth binary mask for visualization.
    """
    random_index = np.random.randint(0, len(xv))
    # Reshape the input image to match the model's input shape
    input_image = np.reshape(xv[14], (1, 512, 512, 1))  # Adjust the shape if needed

    # Make a prediction using the model
    predictions = model.predict(input_image)

    # Assuming predictions and ground truth are in the range [0, 1]
    # Reshape predictions and ground truth if necessary
    predictions = np.reshape(predictions, (512, 512))
    ground_truth = np.reshape(yv[14], (512, 512))

    # Visualize the input image, ground truth, and predictions
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title(input_image)
    plt.imshow(input_image[0, :, :, 0], cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title('Ground Truth')
    plt.imshow(ground_truth, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title('Model Prediction')
    plt.imshow(predictions, cmap='gray')

    plt.show()


def visualize_predictions_outside(unet_model, images):

    predictions = unet_model.predict(images)
    num_images = min(len(images), 12)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 2 * num_images))

    for i in range(4):
        axes[i, 0].imshow(images[i], cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(predictions[i, :, :, 0], cmap='gray')
        axes[i, 1].set_title('Predicted Image')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

input_shape = (512,512,1)
unet = unet_model(input_shape)

initial_learning_rate = 0.0001
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
)
#opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule,clipvalue=1.0)

opt = Adam(learning_rate=0.001 ,clipvalue=1.0)
unet.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

num_samples = 2500 #Specify the number of samples you want to use for training
images, labels = load_samples(image_paths, label_paths, num_samples)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
unet.fit(X_train, y_train, batch_size=5, epochs=10, validation_data=(X_val, y_val)) #epochs
unet.save('D:\proj\model')
#loaded_model = tf.keras.models.load_model('path_to_save_model')s
#predictions = loaded_model.predict(input_data)

#path = r"D:\proj\real_integrals"
#real_integrals = read_and_normalize_images(path)
#visualize_predictions_outside(unet,real_integrals)
visualize_predictions(unet,images,labels)

