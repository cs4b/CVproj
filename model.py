import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras import layers, models
#from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
from keras.models import load_model
import random


class DataLoader:
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_paths = []
        self.label_paths = []
        self.load_data()

    def load_data(self):
        for dir in os.listdir(image_dir):
            image_subdir = os.path.join(image_dir, dir)
            if os.path.isdir(image_subdir):  # Check if it's a directory
                # Take only the one image file from samples the directory
                image_files = [f for f in os.listdir(image_subdir) if os.path.isfile(os.path.join(image_subdir, f))]
                if image_files:  # Check if there are any files in the directory
                    image_path = os.path.join(image_subdir, image_files[random.randint(0,3)])
                    self.image_paths.append(image_path)

                    id = dir.replace("integrals_", "")
                    label_name = f"{id}_GT_pose_0_thermal.png"
                    label_path = os.path.join(label_dir, label_name)
                    self.label_paths.append(label_path)

    def load_samples(self, num_samples):
        images = []
        labels = []

        for image_path, label_path in zip(self.image_paths[:num_samples], self.label_paths[:num_samples]):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image as grayscale
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Load label as grayscale

            if image is None or label is None:
                print(
                    f"Warning: Could not read image or label file at {image_path} or {label_path}. Skipping this sample.")
                continue
            # Resize images and labels to a consistent size
            image = cv2.resize(image, (512, 512))
            label = cv2.resize(label, (512, 512))

            images.append(image)
            labels.append(label)

        images_normalized = [image / 255.0 for image in images]
        labels_normalized = [label / 255.0 for label in labels]
        threshold = 0.9
        binary_mask = tf.where(np.array(labels_normalized) > threshold, 1, 0)

        return np.array(images_normalized), np.array(binary_mask)

    def read_and_normalize_images(self, folder_path):
        normalized_images = []

        for filename in os.listdir(folder_path):
            if filename.endswith('.png'):
                image_path = os.path.join(folder_path, filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                normalized_img = img.astype('float32') / 255.0
                normalized_images.append(normalized_img)

        return np.array(normalized_images)


class UNetModel:
    def __init__(self, input_shape):
        self.model = self.build_unet_model(input_shape)
        self.optimizer = self.optimizer()
        self.compile_model()

    def optimizer(self):
        optimizer = Adam(learning_rate=0.0001 ,clipvalue=1.0)
        return optimizer
    def build_unet_model(self, input_shape):
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

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=[self.psnr_metric])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=5):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val))

    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self,path):
        loaded_model = load_model(path)
        self.model.set_weights(loaded_model.get_weights())
        return loaded_model

    @staticmethod
    def psnr_metric(y_true, y_pred):
        # Clip pixel values to be in the range [0, 1]
        y_true = tf.clip_by_value(y_true, 0.0, 1.0)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

        # Compute MSE (Mean Squared Error)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Compute PSNR
        max_pixel_value = 1.0
        psnr = 20 * tf.math.log(max_pixel_value / tf.math.sqrt(mse)) / tf.math.log(10.0)

        return psnr
    def visualize_predictions(self, xv, yv, num):
        """
        Visualizes the input image, ground truth, and model predictions.

        Parameters:
        - model: The trained U-Net model.
        - input_image: The input image for visualization.
        - ground_truth: The ground truth binary mask for visualization.
        """

        ##ground truth in line 152 is commented.

        random_index = np.random.randint(0, len(xv))
        # Reshape the input image to match the model's input shape
        input_image = np.reshape(xv[num-1], (1, 512, 512, 1))  # Adjust the shape if needed

        # Make a prediction using the model
        predictions = self.model.predict(input_image)

        # Assuming predictions and ground truth are in the range [0, 1]
        # Reshape predictions and ground truth if necessary
        predictions = np.reshape(predictions, (512, 512))
        ground_truth = np.reshape(yv[num-1], (512, 512))

        # Visualize the input image, ground truth, and predictions
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title('Input_image')
        plt.imshow(input_image[0, :, :, 0], cmap='gray')

        plt.subplot(1, 3, 2)
        plt.title('Ground Truth')
        plt.imshow(ground_truth, cmap='gray')

        plt.subplot(1, 3, 3)
        plt.title('Model Prediction')
        plt.imshow(predictions, cmap='gray')

        plt.show()

    def visualize_predictions_outside(self, images):
        predictions = self.model.predict(images)
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
class Trainer:
    def __init__(self, data_loader, unet_model):
        self.data_loader = data_loader
        self.unet_model = unet_model
        self.train_loss_history = []
        self.val_loss_history = []

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

    def train_model(self, num_samples, epochs=10, batch_size=5):
        images, labels = self.data_loader.load_samples(num_samples)
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
        self.unet_model.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
        '''
        for epoch in range(epochs):
            history = self.unet_model.model.fit(X_train, y_train, batch_size=batch_size, epochs=1,
                                                validation_data=(X_val, y_val))

            # Append the training and validation loss to the history
            self.train_loss_history.append(history.history['loss'][0])
            self.val_loss_history.append(history.history['val_loss'][0])

            # Plot the training and validation loss
        self.plot_loss()
        self.unet_model.save_model('D:\proj\model')
        '''
    def plot_loss(self):
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

class Tester:
    def __init__(self, unet_model, data_loader):
        self.unet_model = unet_model
        self.data_loader = data_loader

    def test_model(self, test_path):
        real_integrals = self.data_loader.read_and_normalize_images(test_path)
        self.unet_model.visualize_predictions_outside(real_integrals)

    def test_model_focal(self, num):
        images, label = self.data_loader.load_samples(num)
        self.unet_model.visualize_predictions(images,label,num)

    #Not used
    def test_singular(self,num):
        image, label = self.data_loader.load_samples(num)
        self.unet_model.visualize_predictions()

if __name__ == "__main__":
    image_dir = 'D:\proj\separated\integraltest\integrals'
    label_dir = 'D:\proj\separated\GT'

    data_loader = DataLoader(image_dir, label_dir)
    input_shape = (512, 512, 1)
    unet_model = UNetModel(input_shape)
    trainer = Trainer(data_loader, unet_model)
    tester = Tester(unet_model, data_loader)

    # Train the model
    trainer.train_model(num_samples=3000, epochs=10, batch_size=6)

    #Save the model
    unet_model.save_model('D:\proj\model_separated')

    # The tester class can be used from here, because the DataLoader is already instantiated,
    # but the test.py is better written from general prediction
    '''
    # Test the model
    test_path = r"D:\proj\to_train\testfolder"
    tester.test_model(test_path)
    #tester.test_model_focal(16)

    #for i in range(461,475,1):
    #    tester.test_model_focal(i)
    '''