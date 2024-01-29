# CVproj - Restoration of Integral Images
Everything related to Team-C2 CV project

# Req Versions
Python version >= 3.9.0
Pytorch >= 2.0
Tensorflow >= 2.10.0 
keras >= 2.10.0 	


# classes

model.py
	The main model. Contains classes (additionally to the model class) to load, process input data, and visualise predictions. The visualisation is later moved out of the model class and not used in the test.py
 
	Dataloader class:
		Loads and processes (binary masking, normalization) input data.
  
	UNetModel class:
		This is the model. All the related (pre-written)functions are defined in order to be able to parametrise training. The visualisation functions are not used in the test script because they need the custom UNetModel to be loaded, which requrires the Dataloader to be loaded, that in its constructor loads all the samples. It is not very well written.
  
	Trainer class:
		Set the GPU to be used, and parametrising the unet training in order to save train_loss history and val_loss history. Due to the complexity of the network in order to log the history more memory is needed, which means lowering sample number, which in return would dampen the training efficiency. Unfortunately we cannot use it due to resource deficiency.
  
	Main: 
	image_dir = The directory where the integral images are that are used for training.
	label_dir = The ground_truth library. The class can make groups of 4 focal images(focal stack) and the corresponding ground truth.

Test.py
	This file contains two essential functions that has been pulled from the tester class to easily visualise and save predictions. Two visualise functions: visualize_predictions_outside() does not save the predictions, visualize_predictions_and_save() saves them.
	The visualisation should be used with low number of images, or if we want to predict with a lot of images, then the plot should be commented out from the function visualize_predictions_and_save, because it plots in only 1 row.
	
 	model_path = where the model is saved with keras.model
	test_path = where the test_set is
	save_path = where the predictions will be saved.

clean.py
	A script that searches for incomplete sets of iterations (missing input data label or text file) in the provided thermal images based on filename. - Same purpose as Simulation_Cleaner_Abrar.py

seperate.py
	A script to seperate the ground truth, parameter file and the 11 images to run for AOS integrator. 

combine.py (optional)
	A script to pull input data and labels into one (training) directory.
	
graph.py
	A plot to draw loss/iteration graphs.
	
ModelAllImagesSameDirectory.py
	An enhanced Unet Model which takes All images from same directory (4 integrals and a ground truth) for training. Additional Work to check for all focal stack.
 
testtfunet.py
	Test file to run ModelAllImagesSameDirectory.py file by just adding the directory address. 
	
Simulation_Cleaner_Abrar.py
	A script that cleans the originally provided sets of iterations. - Same purpose as clean.py

# How to Use

Follow these steps to use the CVproj project:

## Step 1: Setup and Configuration

1. Ensure you have Python version 3.9 installed on your system.
2. Install the required dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```
3. Set up the directories and configurations in the `model.py` file:
    - Adjust the directory paths for `image_dir` and `label_dir`.
    - Specify the save location for your model.

## Step 2: Training

1. Run the `model.py` script to train the model:
    ```bash
    python model.py
    ```
2. Adjust the training parameters such as epochs, batch size, and num_samples as needed.

## Step 3: Testing

1. Once training is complete, navigate to the `test.py` file.
2. Set the test directory containing the images you want to test on the trained model.
3. Optionally, set the visualization option to visualize results.
4. Run the `test.py` script to perform testing:
    ```bash
    python test.py
    ```

## Optional Steps:

- For enhanced functionality with all images from the same directory, consider using `ModelAllImagesSameDirectory.py`.
- Use `testtfunet.py` to run `ModelAllImagesSameDirectory.py` by just adding the directory address.

Ensure to review the documentation and comments within the code for additional details and customization options.

