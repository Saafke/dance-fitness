# Dance-Fitness

This repository contains two major components. 

1. Code to run 5 3D Human Pose Estimators on a single, or multiple RGB video(s). It then renders the results - i.e. the 3D mesh or 3D skeleton - as follows.

/// Insert Video ///

2. Code to compare two 3D human meshes or skeletons - i.e. comparing the discrepancies in the angles of the limbs - and visualising the discrepancies as follows.

/// Insert Video ///

### Hardware

This code has been tested on an Ubuntu 18.04 machine.


## Testing 5 Human Pose Estimators

1. Clone these repositories and follow their installation instructions.

- ROMP https://github.com/Saafke/ROMP 
- DecoMR https://github.com/Saafke/DecoMR
- ExPose https://github.com/Saafke/expose
- VIBE https://github.com/Saafke/VIBE
- VideoPose3D https://github.com/facebookresearch/VideoPose3D 

2. Now we can run the above methods on your input video(s). Use bash to run the "run50clips.sh" script. Make sure to change the directories in this script, to your correct directories. That is, the directories where you installed the above methods. Execute the following command:

`$ bash run50clips.sh` 

This will run the methods on your input video(s) and store the results - i.e. the estimated 3D meshes (or skeletons) and camera parameters - in the output folders.

3. If we want to visualise the results, we need to render them. This code will visualise the 3D meshes (or skeletons) from a front-view and a side-view (90 degrees rotated about the up-axis) on a white background. It renders the estimations via a weak-perspective camera model. [Note: the results for VideoPose3D have already been rendered, so need for that here.] Execute the following command:

`$ python my-renderer.py`

4. The above code renders independent videos for each method. To combine the videos - i.e. visualise them into 3 columns and 2 rows, as shown above - we can use the make_qual.py file. Again, make sure to change the directories.

`$ python make_qual.py`


## Comparing Two Human Meshes/Skeletons