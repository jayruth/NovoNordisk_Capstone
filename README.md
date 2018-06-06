# NovoNordisk_Capstone
This repository contains the source code for our DIRECT capstone project with Novo Nordisk.
Our goal is to predict protein expression from nucleotide/amino acid sequence using the 
data set from Sastry et el. Their source code and original data files can be found 
at https://github.com/SBRG/Protein_ML.

# NOVO-DNA/AA-Classfication (NDAC): Multiclassification of Protein Characteristics from Nucleotide or Amino Acid Sequences 

**Download Software**

In command line, type: pip install Novo-dna-aa

**Software Description**

SAFMI is a software for the identification of different textural components and their segmentation in AFM images of self assembled peptide structures. From a .txt file image data of the system from AFM, the software helps front-end users approximate the effect of natural tip drift which is ubiquitous in AFM characterization, cancel noise, detect different textures based on how ordered or disordered the self-assembly is and calculate the percent coverage of each texture, overall percent coverage of the self assembled peptides, and the ordered to disordered ratio as a means of characterizing the results in terms of degree of disorderedness. This will help evaluate the self assembly properties of the peptides on the substrates for the processing conditions.
A second use case of SAFMI is to output the same data even without the availability of an AFM image, if the processing parameters such as pH and peptide-concentration are known. A k-Nearest Neighbors classification algorithm is used for a regression analysis of the image parameters (percent coverage, Ratio of Ordered to Disordered, degree of disorderedness) from 55 512x512 pixel size AFM images of such systems, 75% of which are used for training and 25% for validation. This technique predicts the estimated Ratio of Ordered to Disordered for an absent image if the processing conditions are known.

**Software Dependencies:**

Python version 3

Python Packages: Scikit-Image, OpenCV, Scikit-Learn, Matplotlib, Numpy, Scipy and Pandas

Mac OS X and Windows are both able to download python 3 without any dependencies. a `conda install "your library package"` code will need to be run on the terminal or git-Bash in order to install numpy, scipy, pandas, mATPLOTLIB, Scikit-Learn, Scikit-Image and openCV. Familiarity with conda, sudo and pip commands will be useful for properly downloading and installing these packages.

**Directories:**

* *UserImageProcessing:*
  - This folder contains all the functions needed for preprocessing of the AFM image for the first use case in which the AFM image is available.

  - To run the software, either use `pip install SAFMI_AFM` or git clone the whole repository into your computer.

  - In a shell script, either terminal for MacOSX or any bash enabled command line GUI in windows, open the directory "UserImageProcessing" by running `cd /path/to/cloned/repository/UserImageProcessing`, type in and run: python UserImageProcessing.py

  - This interactive software will then provide you with instructions on how to load, preprocess and segment your AFM raw.txt file. Errors, if any, will be displayed, and possible causes displayed as well, with hints for troubleshooting.
  You will be able to see a grayscale version of your image, and you will be able to choose between background removal, image segmentation and image separation into different texture/phases depending on what you want.

  - If you choose `Background Removal` you will be able to choose between a rectangular, square and disk shaped background approximation to estimate tip drift, based on how your loaded image looks. At every step your input will be asked for to provide you with an opportunity to go back and play around with the different background removal options until you are satisfied with how your image looks. Errors if any in the user-inputs will be displayed and hints for troubleshooting provided.
  It will then be sent for noise cancellation and normalization to preview a corrected grayscale image.

  - If you choose `Segmentation`, you will be able to play around with background correction, until a satisfactory grayscale normalized image is displayed. Then you will be able to choose a threshold value to perform image segmentation to separate out the ordered and disordered regions based on a random walker algorithm. you can change the threshold until you are satisfied and then the software will show you a segmented version of your image. Errors if any in the user-inputs will be displayed and hints for troubleshooting provided.

  - If you choose `Separation`, the image will be background corrected, segmented and then separated into two different images based on a similar interactive iterative process where you can play around with the parameters of the separation function until you are satisfied. At the end you will be able to get a numerical estimate of the Order to Disorder ratio, percent surface coverage of each texture and overall percent coverage of the peptides. Errors if any in the user-inputs will be displayed and hints for troubleshooting provided.

  - The directory sample_images contains raw peptide AFM images.

* *ImageDisorderPrediction*
  - This folder contains all the functions needed for predicting the Ratio of order to disorder, or the degree of disorderedness of peptide self-assembly when a relevant AFM image is not available with the user, provided the user has information about the pH and concentration of peptide that they want to use.
  - To run the software, first git clone the whole repository into your computer.

  - In a shell script, either terminal for MacOSX or any bash enabled command line GUI in windows, open the directory "UserImageProcessing" by running `cd /path/to/cloned/repository/ImageDisorderPrediction`, type in and run: python UserPrediction.py

  - This interactive software will then provide you with instructions on how to enter pH (between 1 and 14) and concentration of peptide solution (between 0.1 to 2 micro-Molar (uM)).
    You will be able to manipulate iteratively the k value of a k-nearest neighbors algorithm to predict a degree of disorder "*highly disordered, completely disordered or completely ordered*" for the set of processing conditions that you enter. Errors if any in the user-inputs will be displayed and hints for troubleshooting provided.

  - The knn_test.py splits a set of 55 processed AFM images to training and testing sets (75% training). Prediction results of the testing set are plotted. Errors are calculated by comparing the disorder level of the prediction and the test data. The file containing the data set is in the same directory, and named 'afm_datafile_v3.csv'.

  - Prediction of the test data described above is *~73%*. Visualization of test-set prediction are saved in this directory: Order_Disorder_ratio.png and testingdata_vs_prediction.png.




#### All the functions used in the software use cases are available to play with in their respective folders.

*For running tests:*
In a shell script, open the directory "UnitTests", type in and run: nosetests -verbose Test.(the name of 'test.py' file desired to run) based on the use cases you want to access.
