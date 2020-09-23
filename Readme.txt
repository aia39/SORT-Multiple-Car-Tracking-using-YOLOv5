
##############################################################################
		Instructions for Running Tracker on Images
##############################################################################


==============================================================================
|1|			Installing Dependencies
==============================================================================

(You need anaconda installed in your device to execute this)
From a terminal on linux/macOS or a cmd.exe on windows machine:

a) Creating new conda environment:

	> conda create –n myenv python=3.7
	> conda activate myenv

b) Install python dependencies:

	> conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
	> conda install -yc anaconda numpy opencv matplotlib tqdm pillow ipython
	> conda install -yc conda-forge scikit-image 
	> conda install -c anaconda scikit-learn
	> conda install -yc conda-forge protobuf numpy
	> conda install -c conda-forge filterpy
	> conda install -c anaconda scipy
	> pip install numba

# Alternative : The above installations can be also done using the "track_requirements.txt" file:
	
	> pip install -U -r track_requirements.txt



==============================================================================
|2|			Setting up workspace directory
==============================================================================

Keep the image frames of one video/sequence in a folder so that the tracker can run on a video sequence.
The inferred output will be on that single sequence. If you don't want consistency in sequences then 
put all images in a folder.
 
For example if 'images' folder is used as input folder of a video in the parent directory. The full path to
the image folder will be like this: 

	E:\VIP2020\Tracker Final\images
	
	
Multiple folders in the parent directory can be made for multiple videos output.
 

==============================================================================
|3|				Executing tracker
==============================================================================

Run the 'main.py' in command shell to generate video of realtime tracking. 



Please correctly specify all arguments before running the code.



#--- SETTING INPUT IMAGE DIRECTORY ---#
 
You can specify input folder for each run by entering “python main.py --inp images”
if input folder is ‘images’ or enter your desired input folder name.
	
	> python main.py --inp images



#--- OUTPUT VIDEO FILE ---#

Output video will be saved into a folder named “output” in the parent directory. 
You can specify the name of output video by entering “python main.py --inp images --out output_video.mp4”.
Video will be saved at 20 FPS by default

	> python main.py --inp images --out output_video.mp4



#--- INPUT IMAGE FORMATS ---#

If your input directory image has format other than .jpg then please mention this in argument by entering 
“python main.py --inp images –i_formt png” if image has .png format.

	> python main.py --inp images –i_formt png



#--- SPECIAL NOTE ---#

Make sure to change output video name each time if you run it in multiple input folders for separate video. 
Otherwise it will overwrite the previous output video.



