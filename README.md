# CVproj
Anything related to C2 CV project


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

AOSIntegratorTry3.py was not used.

clean.py
	A script that searches for incomplete sets of iterations (missing input data label or text file) in the provided thermal images based on filename. - Same purpose as Simulation_Cleaner_Abrar.py

combine.py 
	A script to pull input data and labels into one (training) directory.
	
graph.py
	A plot to draw loss/iteration graphs.
	
*ModelAllImagesSameDirectory.py

*testtfunet.py

separate.py
	A script to move around input data and labels to two different directories.
	
Simulation_Cleaner_Abrar.py
	A script that cleans the originally provided sets of iterations. - Same purpose as clean.py


