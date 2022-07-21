#!/usr/bin/env python
#
#
# 3D Human Pose Estimation and Evaluation
# Copyright (C) 2022  Xavier Weber

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
#
# This is a script to run the five 3D Human Pose Estimation methods
# Note: 
# - You have to change all of the directories in this script, to YOUR own local directories.
# - You might have to change the "/home/anaconda3/etc/profile.d/conda.sh" paths to your own anaconda folder.

### ADD YOUR DIRECTORY PATHS
CLIPS_DIR=""
RESULTS_DIR=""

VIBE_DIR = "" 
VIBE_ENV_DIR = ""
VIBE_STAF_DIR = ""

EXPOSE_DIR = ""
EXPOSE_ENV_DIR = ""

ROMP_DIR = ""
ROMP_ENV_DIR = ""

DECOMR_DIR = ""
DECOMR_ENV_DIR = ""
OPEN_POSE_DIR = ""

VIDEOPOSE3D_DIR = ""
VIDEOPOSE3D_ENV_DIR = ""

# Run flags
RUN_VIBE=true
RUN_EXPOSE=true
RUN_ROMP=true
RUN_DECOMR=true
RUN_VIDEOPOSE3D=true


################################################################
# Go to VIBE directory and activate corresponding conda env
if ($RUN_VIBE); then

    cd $VIBE_DIR
    . /home/anaconda3/etc/profile.d/conda.sh && conda activate $VIBE_ENV_DIR

    for i in {00..}
    do
    echo "${CLIPS_DIR}$i.mp4"
    python demo.py --tracking_method pose --run_smplify --staf_dir $VIBE_STAF_DIR \
                --vid_file "${CLIPS_DIR}$i.mp4" \
                --output_folder "./output/vibe" \
                --no_render
    done
fi


################################################################
# Go to ExPose directory and activate corresponding conda env
if ($RUN_EXPOSE); then
    cd $EXPOSE_DIR
    . /home/anaconda3/etc/profile.d/conda.sh && conda activate $EXPOSE_ENV_DIR

    for i in {00..}
    do
    echo "${CLIPS_DIR}$i.mp4"
    python3 demo.py --video-path "${CLIPS_DIR}$i.mp4" --exp-cfg data/conf.yaml \
                    --show=False --output-folder "./output/expose" \
                    --save-params False --save-vis False --save-mesh True
    done
fi


###############################################################
#Go to ROMP directory and activate corresponding conda env
if ($RUN_ROMP); then
    cd $ROMP_DIR
    . /home/anaconda3/etc/profile.d/conda.sh && conda activate $ROMP_ENV_DIR

    for i in {00..}
    do 
    # Update src/configs/video.yml (input video and output folder!) LINE 20
    sed -i "20s|.*| input_video_path: ${CLIPS_DIR}$i.mp4|" "$ROMP_DIR/src/configs/video.yml"
    sed -i "21s|.*| output_dir: ./output/romp/$i|" "$ROMP_DIR/src/configs/video.yml"

    # ROMP run command
    CUDA_VISIBLE_DEVICES=0 python3 core/test.py --gpu=0 --configs_yml=configs/video.yml
    done
fi


################################################################
# Go to the OpenPose directory and run the detections (we need them for DecoMR)
if ($RUN_DECOMR); then
    
	cd $OPENPOSE_DIR

    for i in {00..}
    do 
        # Make directory to store keypoint detections
        mkdir -p "output/openpose/$i"

        # Run openpose on the images
        ./build/examples/openpose/openpose.bin\
        --write_json "output/openpose/$i"\
        --video "${CLIPS_DIR}$i.mp4"
    done

    # #Go to DecoMR directory and activate corresponding conda env
    cd $DECOMR_DIR
    . /home/anaconda3/etc/profile.d/conda.sh && conda activate $DECOMR_ENV_DIR
    
	for i in {00..}
    do  
        # Make folder to store images
        mkdir -p "${CLIPS_DIR}$i.mp4" "${DECOMR_DIR}images/$i"

        # Convert video to images first
        ffmpeg -i "${CLIPS_DIR}$i.mp4" "${DECOMR_DIR}images/$i/%06d.png"

        # Run DECOMR on the images
        python demo.py --checkpoint=data/model/h36m_up3d/checkpoints/h36m_up3d.pt\
        --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json\
        --config=data/model/h36m_up3d/config.json\
        --image_folder "${DECOMR_DIR}images/$i"\
        --openpose_video "${DECOMR_DIR}openpose-results/$i"\
        --output_folder "./output/decomr/$i"\
        --no_render
    done
fi


################################################################
if ($RUN_VIDEOPOSE3D); then
    
    # Go to the VideoPose3D directory and activate the conda env
    cd $VIDEOPOSE3D_DIR

    python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir "./output/videopose3d" --image-ext mp4 "$CLIPS_DIR"

    # Go into data folder
    cd ../data

    python prepare_data_2d_custom.py\
    -i "./output/videopose3d" -o myvideos

    cd ..

    for i in {00..}
    do  
        python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint\
        --evaluate pretrained_h36m_detectron_coco.bin --render\
        --viz-subject "$i.mp4" --viz-action custom --viz-camera 0\
        --viz-video "${CLIPS_DIR}$i.mp4"\
        --viz-output output.mp4 --viz-size 6
    done
fi