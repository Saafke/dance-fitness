#!/usr/bin/env python
#
################################################################################## 
# Authors: 
#   Xavier Weber: eey138@qmul.ac.uk
#
#  Created Date: 2022/07/15
#
#####################################################################################
# MIT License
#
# Copyright (c) 2022 Xavier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#####################################################################################
#
# This is a script to run the five 3D Human Pose Estimation methods
# Note: you have to change all of the directories in this script, to YOUR local directories

# Update temp directory
export TMPDIR=/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/tmp

# Input clips and root of output directory 
CLIPS_DIR='/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/CLIPS/side/'
RESULTS_DIR='/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS2/'


RUN_VIBE=true
RUN_EXPOSE=true
RUN_ROMP=true
RUN_DECOMR=true
RUN_VIDEOPOSE3D=true


################################################################
# Go to VIBE directory and activate corresponding conda env
if ($RUN_VIBE); then

    cd /media/weber/Ubuntu2/ubuntu2/Human_Pose/code-from-source/VIBE
    . /home/weber/anaconda3/etc/profile.d/conda.sh && conda activate /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/envs/env_VIBE

    # If above don't work, do this manually in the terminal where you will execute this file
    #source /Users/weber/anaconda/bin/activate /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/envs/env_VIBE
    #conda activate /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/envs/env_VIBE

    for i in {00..50}
    do
    echo "${CLIPS_DIR}$i.mp4"
    python demo.py --tracking_method pose --run_smplify --staf_dir /home/weber/Documents/from-source/VIBE/STAF \
                --vid_file "${CLIPS_DIR}$i.mp4" \
                --output_folder "${RESULTS_DIR}vibe" \
                --no_render
    done
fi


################################################################
# Go to ExPose directory and activate corresponding conda env
if ($RUN_EXPOSE); then
    cd /media/weber/Ubuntu2/ubuntu2/Human_Pose/code-from-source/expose
    . /home/weber/anaconda3/etc/profile.d/conda.sh && conda activate /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/envs/expose

    for i in {00..50}
    do
    echo "${CLIPS_DIR}$i.mp4"
    python3 demo.py --video-path "${CLIPS_DIR}$i.mp4" --exp-cfg data/conf.yaml \
                    --show=False --output-folder "${RESULTS_DIR}expose" \
                    --save-params False --save-vis False --save-mesh True
    done
fi


###############################################################
#Go to ROMP directory and activate corresponding conda env
if ($RUN_ROMP); then
    cd /media/weber/Ubuntu2/ubuntu2/Human_Pose/code-from-source/ROMP/src
    . /home/weber/anaconda3/etc/profile.d/conda.sh && conda activate /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/envs/romp

    for i in {00..50}
    do 
    # Update src/configs/video.yml (input video and output folder!) LINE 20
    sed -i "20s|.*| input_video_path: ${CLIPS_DIR}$i.mp4|" /media/weber/Ubuntu2/ubuntu2/Human_Pose/code-from-source/ROMP/src/configs/video.yml
    sed -i "21s|.*| output_dir: ${RESULTS_DIR}romp/$i|" /media/weber/Ubuntu2/ubuntu2/Human_Pose/code-from-source/ROMP/src/configs/video.yml

    # ROMP run command
    CUDA_VISIBLE_DEVICES=0 python3 core/test.py --gpu=0 --configs_yml=configs/video.yml
    done
fi


################################################################
# Go to the OpenPose directory and run the detections (we need them for DecoMR)
if ($RUN_DECOMR); then
    # cd /home/weber/Documents/from-source/openpose

    # for i in {00..50}
    # do 
    #     # Make directory to store keypoint detections
    #     mkdir -p "/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS2/decomr/openpose-results/$i"

    #     # Run openpose on the images
    #     ./build/examples/openpose/openpose.bin\
    #     --write_json "/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS2/decomr/openpose-results/$i"\
    #     --video "${CLIPS_DIR}$i.mp4"
    # done

    # #Go to DecoMR directory and activate corresponding conda env
    cd /media/weber/Ubuntu2/ubuntu2/Human_Pose/code-from-source/DecoMR
    . /home/weber/anaconda3/etc/profile.d/conda.sh && conda activate /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/envs/decomr

    for i in {23..50}
    do  
        # Root folder
        DECO_DIR="/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS2/decomr/"
        DECO_DIR2="/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS/decomr/"
        
        # Make folder to store images
        mkdir -p "${CLIPS_DIR}$i.mp4" "${DECO_DIR}images/$i"

        # Convert video to images first
        ffmpeg -i "${CLIPS_DIR}$i.mp4" "${DECO_DIR}images/$i/%06d.png"

        # Run DECOMR on the images
        python demo.py --checkpoint=data/model/h36m_up3d/checkpoints/h36m_up3d.pt\
        --img=examples/im1010.jpg --openpose=examples/im1010_openpose.json\
        --config=data/model/h36m_up3d/config.json\
        --image_folder "${DECO_DIR}images/$i"\
        --openpose_video "${DECO_DIR2}openpose-results/$i"\
        --output_folder "${DECO_DIR}deco-results/$i"\
        --no_render
    done
fi


################################################################
if ($RUN_VIDEOPOSE3D); then
    
	# Deactivate any conda env
    conda deactivate

    # Go to the VideoPose3D directory and activate the conda env
    cd /media/weber/Ubuntu2/ubuntu2/Human_Pose/code-from-source/VideoPose3D/inference

    python infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS2/videopose3d/after-inference --image-ext mp4 "$CLIPS_DIR"

    # Go into data folder
    cd ../data

    python prepare_data_2d_custom.py\
    -i /mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS2/videopose3d/after-inference -o myvideos

    cd ..

    for i in {00..50}
    do  
        python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint\
        --evaluate pretrained_h36m_detectron_coco.bin --render\
        --viz-subject "$i.mp4" --viz-action custom --viz-camera 0\
        --viz-video "${CLIPS_DIR}$i.mp4"\
        --viz-output output.mp4 --viz-size 6
    done
fi