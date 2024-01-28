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
print("gpu", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=18000)]
            )
    except RuntimeError as e:
        print(e)


data_dir = r'D:\proj\to_train'
image_paths = []
label_paths = []

image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
print("total simulations: ", len(image_files) / 5)
for i in range(0, len(image_files), 5):
    label_path = os.path.join(data_dir, image_files[i])
    if i + 4 >= len(image_files):
        print(f"Index out of range: i={i}, i+4={i + 4}, len(image_files)={len(image_files)}")
        continue
    # Take the first four images as input and the fifth as label
    input_images = [os.path.join(data_dir, image_files[i+1 + j]) for j in range(4)]
    #label_path = os.path.join(data_dir, image_files[i + 4])

    image_paths.append(input_images)
    label_paths.append(label_path)
'''
for i in range(min(5, len(image_paths))):
    print(f"Input Images {i+1}: {image_paths[i]}")
    print(f"Label Path {i+1}: {label_paths[i]}")
    print()

num_samples_to_visualize = min(5, len(image_paths))

for i in range(num_samples_to_visualize):
    input_images_sample = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths[i]]
    concatenated_image_sample = np.mean(input_images_sample, axis=0)
    label_sample = cv2.imread(label_paths[i], cv2.IMREAD_GRAYSCALE)

    # Visualize the input images, concatenated image, and label
    plt.figure(figsize=(16, 4))


    plt.subplot(1, 5, 5)
    plt.title('Concatenated Image & Label')
    plt.imshow(np.hstack((concatenated_image_sample, label_sample)), cmap='gray')
    plt.axis('off')

    for j in range(4):
        plt.subplot(1, 5, j + 1)
        plt.title(f'Input Image {j + 1}')
        plt.imshow(input_images_sample[j], cmap='gray')
        plt.axis('off')

    plt.show()
'''
def load_samples(image_paths, label_paths, num_samples):
    images = []
    labels = []

    for input_images, label_path in zip(image_paths[:num_samples], label_paths[:num_samples]):
        # Load and concatenate the four input images
        input_images_data = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in input_images]
        input_images_avg = np.mean(input_images_data, axis=0)

        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Load label as grayscale

        if input_images_avg is None or label is None:
            print(f"Warning: Could not read input images or label file at {input_images} or {label_path}. Skipping this sample.")
            continue

        # Resize images and label to a consistent size
        input_images_avg = cv2.resize(input_images_avg, (512, 512))
        label = cv2.resize(label, (512, 512))

        images.append(input_images_avg)
        labels.append(label)

    images_normalized = [image / 255.0 for image in images]
    labels_normalized = [label / 255.0 for label in labels]
    threshold = 0.88
    binary_mask = tf.where(np.array(labels_normalized) > threshold, 1, 0)

    return np.array(images_normalized), np.array(binary_mask)


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

input_shape = (512,512,1)
unet = unet_model(input_shape)

initial_learning_rate = 0.01
lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=False
)

opt = Adam(learning_rate=lr_schedule, clipvalue=1.0)
unet.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

num_samples = 1700  #Specify the number of samples you want to use for training
images, labels = load_samples(image_paths, label_paths, num_samples)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.05, random_state=30)
unet.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_val, y_val)) #epochs
unet.save('F:\CvDataset\Dataset\modelNEW')
