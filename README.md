## 3D Human Body
This repository contains two major components. 

1. Run five 3D-Human-Pose-Estimation methods on a single, or multiple RGB video(s). Render the results - the estimated 3D meshes and 3D skeletons - on top of a white background.

2. Compare two 3D human meshes. That is, compare the discrepancies in the angles of the limbs, and visualise the discrepancies on the mesh as shown below. The colored mesh is the input mesh, which we want to compare with a reference mesh. The limbs of this mesh are colored based on the correctness. Green means that the limb angle is within the accepted error range with respect to the reference, and red means it is outside of the accepted error range.
![Comparing 3D meshes](compare_two_humans/output.png)

#### Hardware

This code has been tested on an Ubuntu 18.04 machine.


# Testing five 3D-Human-Pose-Estimation models

1. Clone these repositories and follow their installation instructions.

| Method      		| Repository link |
| ----------------- | ----------------|
| ROMP [1]    		| [[Link](https://github.com/Saafke/ROMP)]		|
| DecoMR [2]   		| [[Link](https://github.com/Saafke/DecoMR)]  |
| ExPose [3]   		| [[Link](https://github.com/Saafke/expose)]  |
| VIBE [4]   		| [[Link](https://github.com/Saafke/VIBE)]  |
| VideoPose3D [5]   | [[Link](https://github.com/facebookresearch/VideoPose3D )]  |

_Note: The above links for ROMP, DecoMR, ExPose and VIBE are forks and improved over the original, to facilitate running these methods on videos._

2. Now we can run the above methods on your input video(s). Use bash to run the "estimate.sh" script. Make sure to change the directories in this script, to your correct directories. That is, the directories where you installed the above methods. Execute the following command:

First, go into the correct subfolder of this repository:

`$ cd test_five_methods`

Then run:

`$ bash estimate.sh` 

This will run the methods on your input video(s) and store the results - i.e. the estimated 3D meshes (or skeletons) and camera parameters - in the output folders.

3. If we want to visualise the results, we need to render them. The following script will visualise the 3D meshes (or skeletons) from a front-view and a side-view (90 degrees rotated about the up-axis) on a white background. It renders the estimations via a weak-perspective camera model. Execute the following command:

`$ python render.py`

4. The above code renders independent videos for each method. To combine the videos into a single view:

`$ python mix_clips.py`


# Comparing Two Human Meshes

This will compare two estimated SMPL [6] meshes. We first extract the 3D skeleton from the meshes. We compute the discrepancies between skeleton1 and skeleton2, i.e. the differences between the limbs' 3D angles. The error threshold is a hyperparameter you can change. To try the toy-example:

#### Install requirements via conda
```
conda create -n two-humans python=3.8
conda activate two-humans
pip install -r compare_two_humans/requirements.txt
```

#### Execute script

`python compare_two_humans/compare_and_vis.py`

Your result is the output.png file. To try your own meshes, change the file paths in compare_and_vis.py with your own estimations or ground-truths. These should be in the form of SMPL models, such the estimations from VIBE, ExPose or ROMP.

## References
[1] Sun, Y., Bao, Q., Liu, W., Fu, Y., Black, M. J., & Mei, T. (2021). **Monocular, one-stage, regression of multiple 3d people.** In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 11179-11188).[[Weblink](https://openaccess.thecvf.com/content/ICCV2021/html/Sun_Monocular_One-Stage_Regression_of_Multiple_3D_People_ICCV_2021_paper.html)]

[2] Zeng, W., Ouyang, W., Luo, P., Liu, W., & Wang, X. (2020). **3D Human Mesh Regression with Dense Correspondence.** In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7054-7063). [[Weblink](https://openaccess.thecvf.com/content_CVPR_2020/html/Zeng_3D_Human_Mesh_Regression_With_Dense_Correspondence_CVPR_2020_paper.html)]

[3] Choutas, V., Pavlakos, G., Bolkart, T., Tzionas, D., & Black, M. J. (2020, August). **Monocular expressive body regression through body-driven attention.** In European Conference on Computer Vision (pp. 20-40). Springer, Cham. [[Weblink](https://expose.is.tue.mpg.de/)]

[4] Kocabas, M., Athanasiou, N., & Black, M. J. (2020). **Vibe: Video inference for human body pose and shape estimation.** In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 5253-5263). [[Weblink](https://openaccess.thecvf.com/content_CVPR_2020/html/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.html)]

[5] Pavllo, D., Feichtenhofer, C., Grangier, D., & Auli, M. (2019). **3D Human Pose Estimation in Video with Temporal Convolutions and Semi-supervised Training.** In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 7753-7762). [[Weblink](https://openaccess.thecvf.com/content_CVPR_2019/html/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.html)]

[6] Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., & Black, M. J. (2015). **SMPL: A skinned multi-person linear model.** ACM transactions on graphics (TOG), 34(6), 1-16. [[Weblink](https://smpl.is.tue.mpg.de/)]

## Contact

This code has been written by Xavier Weber and Mohamed Ilyes Lakhal. For queries regarding this repository, please contact Xavier (eey138@qmul.ac.uk).