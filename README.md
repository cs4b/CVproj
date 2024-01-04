# CVproj
Anything related to C2 CV project

### clean.py 
script to remove incomplete iterations.

In line 6, add your basedirectory names

In line 13, name your output directory to copy the complete sets. I am not deleting incomplete sets, instead copying complete ones to prevent dataloss in case the script is not working right.

In line 42, 43 the copy command is commented. remove the # to actually copy the complete sets to the output directory.


### separate.py 
script to separate iterations from ground truth and segment the iterations into folders 

last line, pass the folder that has the complete iterations and a folder to save the separated pictures. note both all GT images are saved in 1 folder and each iteration is saved into its folder


### integrate 
script to implement the integrator for all the iterations at various focal planes. 
provide or change focal planes line 166
provide folder containing all iterations (output from separate.py)
🤞🏾 the integrator just refused to work for mac so its not tested yet. but it should 