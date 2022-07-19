#####################################################################################
#
"""
In this script the renderer takes as input:

Mesh:
- Vertices
- Faces

Camera:
- in- and extrinsics

Image:
- input image dimensions

which output white images with two poses (same pose but one normal view, one side view). 
"""

# some code taken from VIBE: https://github.com/mkocabas/VIBE

import math
import trimesh
import pyrender
import numpy as np
import vg
import json
import subprocess
import cv2
import os
import re
from d3g.angle import lineAndPlane, lines
from pyrender.constants import RenderFlags
#from lib.models.smpl import get_smpl_faces

class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=True):
        self.resolution = resolution

        #self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, verts, faces, romp_scaler=1, ht=0, ht_cam=1, deco_cam=None, weak_cam=None, cam=None, angle=None, axis=None, flip=False, color=[1.0, 0.0, 0.0]):
        """
        ht = horizontal translation
        """
        # Create mesh object
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        if flip == True:
            Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
            mesh.apply_transform(Rx)

        # Rotate mesh if necessary
        if angle and axis:

            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)
            #get_all_head_vertex_locations(mesh.vertices) 
            #angle = get_all_head_vertex_locations(mesh.vertices)
            #angle = (angle-90)*-1
            #print("Angle=",angle)
            #R = trimesh.transformations.rotation_matrix(math.radians(angle), [0,0,1])
            #mesh.apply_transform(R)
        #else:
            #get_all_head_vertex_locations(mesh.vertices)
        
        # Add a camera object
        if weak_cam is not None: # weak-perspective camera parameters
            sx, sy, tx, ty = weak_cam
            camera = WeakPerspectiveCamera(
                scale=[sx/romp_scaler, sy],
                translation=[tx+ht, ty],
                zfar=1000.
            )   
            camera_pose = np.eye(4)
            cam_node = self.scene.add(camera, pose=camera_pose, name='weak_camera')
        
        elif deco_cam is not None: # only camera translation is given
            translation = deco_cam 
            focal_length = 5000*1.5
            center = img.shape[1]/2/ht_cam, img.shape[0]/2

            camera = pyrender.IntrinsicsCamera(
                fx=focal_length,
                fy=focal_length,
                cx=center[0],
                cy=center[1],
            )
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = translation.copy()
            camera_pose[0, 3] *= (-1)
            cam_node = self.scene.add(camera, pose=camera_pose, name='pinhole_camera')
        
        elif cam: #pin-hole camera parameters
            focal_length = cam.item().get('focal_length')
            translation = cam.item().get('translation')
            center = cam.item().get('center')

            camera = pyrender.IntrinsicsCamera(
                fx=focal_length,
                fy=focal_length,
                cx=center[0][0]/ht_cam,
                cy=center[0][1],
            )
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = translation.copy()
            camera_pose[0, 3] *= (-1)
            cam_node = self.scene.add(camera, pose=camera_pose, name='pinhole_camera')
        else:
            raise ValueError('Please specify camera parameters for the rendering step.')
        
        # Create the material for the mesh and add it to the scene
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node = self.scene.add(mesh, 'mesh')

        # Rendering options
        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image

def images2video(folder):
    command = "ffmpeg -y -i {}/%06d.png {}/output.mp4".format(folder, folder)
    subprocess.call(command,shell=True)

# Get the vertices from the head and their locations, then figure out the middle point.
def get_all_head_vertex_locations(verts):

    # Load semantic body part labelling dictionary (maps vertex index to semantic body part label)
    f = open('/media/weber/Ubuntu2/ubuntu2/Human_Pose/checker-code/files/part_based_vertex_label.json')
    labels = json.load(f)

    # Get the mean of all 3D points of the head
    head_indices = labels['head']
    head_verts = verts[head_indices]
    mean_of_head = np.mean(head_verts, axis=0)
    print("Mean point of the head", mean_of_head)

    # Get the mean of all 3D points of the hips
    hips_indices = labels['hips']
    hips_verts = verts[hips_indices]
    mean_of_hips = np.mean(hips_verts, axis=0)
    #print("Mean point of the hips", mean_of_hips)

    # Get the line between these two 3D points
    #diff = mean_of_head - mean_of_hips
    
    # Get the angle between the line and the plane
    #angle = lineAndPlane(line = [mean_of_head,diff], plane = [0,0,0,0])
    #angle = lines(line1 = [mean_of_head,diff], line2 = [ [0, 0, 0], [0, 0, 1] ])
    #angle = vg.angle(np.array(diff), np.array([0,0,1]), look=vg.basis.z)
    

    #print("angle=", angle)

    return mean_of_head, mean_of_hips

def get_original_image_size(index):
    path = os.path.join('/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips/RESULTS2/decomr/images/{}/0001.png'.format(index))
    image = cv2.imread(path)
    return image.shape

def render_decomr(index, dir):
    # ----------------------------------------------------------
    # LOAD decomr ESTIMATIONS # these meshes have 6890 vertices
    decomr_meshes_dir = os.path.join(dir, 'RESULTS2/decomr/deco-results/{}/images/meshes'.format(index))
    decomr_cams_dir = os.path.join(dir, 'RESULTS2/decomr/deco-results/{}/images/cams'.format(index))
    
    decomr_meshes_paths = [f for f in os.listdir(decomr_meshes_dir) if f.endswith('.obj')]
    decomr_meshes_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    decomr_cams_paths = [f for f in os.listdir(decomr_cams_dir) if f.endswith('.npy')]
    decomr_cams_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    print("length of meshes_paths:", decomr_meshes_paths)
    print("length of meshes_paths:", decomr_cams_paths)

    out_path = os.path.join(dir, 'render-results', 'decomr', index)
    print(out_path)
    os.makedirs(out_path, exist_ok=True)

    height,width,channels = get_original_image_size(index)

    # Loop over the meshes and render them
    for idx in range(0, len(decomr_meshes_paths)-1):

        print('DecoMR:', idx, end='\r')
        
        # background white image

        # load mesh as trimesh object
        mesh = trimesh.load(os.path.join(decomr_meshes_dir,decomr_meshes_paths[idx]), process=False)

        # load estimated camera info
        orig_shape, ul, camera_translation = np.load(os.path.join(decomr_cams_dir,decomr_cams_paths[idx]), allow_pickle=True)
        
        R = Renderer((224,224))
        white_image = np.ones((224,224,3))*255
        
        R = Renderer((width,height))
        white_image = np.ones((height,width,3))*255

        # render mesh on top of the white image
        render_result = R.render(white_image, mesh.vertices, mesh.faces, ht_cam=2, deco_cam=camera_translation, flip=True)

        # render side mesh on top of the previous result
        render_result = R.render(render_result, mesh.vertices, mesh.faces, angle=270, axis=[0,1,0], ht_cam=0.75, deco_cam=camera_translation, flip=True)

        # Revert back to original image size
        # uncropped_img_render = cv2.resize(render_result, dsize=(orig_shape[1],orig_shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # # DEBUGGING WEIRD CROP SIZES
        # original_img = np.ones((height,width,3))*255
        # temp = 0
        # extra = 0 
        # # final crop pixel indices are below zero: impossible
        # if ul[1] < 0:
        #     temp = -1*ul[1]
        #     print("We will cut off extra from top:", temp)
        #     ul[1] = 0
        #     original_img[ul[1]:ul[1]+uncropped_img_render.shape[0]-temp, ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render[temp:,:,:]

        # # final crop pixel indices go over original image size
        # elif ul[1]+uncropped_img_render.shape[0] > original_img.shape[0]:
        #     extra = ul[1]+uncropped_img_render.shape[0] - original_img.shape[0]
        #     print("We will cut off extra from bottom:", extra)
        #     original_img[ul[1]:ul[1]+uncropped_img_render.shape[0]-extra, ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render[:-extra,:,:]

        # else:
        #     original_img[ul[1]:ul[1]+uncropped_img_render.shape[0], ul[0]:ul[0]+uncropped_img_render.shape[1],:] = uncropped_img_render
        
        cv2.imwrite( os.path.join(out_path, '{:06d}.png'.format(idx)), render_result)
        #cv2.imwrite( os.path.join(out_path, '{:06d}.png'.format(idx)), original_img)

    images2video(out_path)

def render_romp(index, dir):
    # ----------------------------------------------------------
    # LOAD ROMP ESTIMATIONS # these meshes have 6890 vertices
    romp_meshes_dir = os.path.join(dir,'RESULTS2/romp/{}'.format(index))
    #print(dir, romp_meshes_dir)
    romp_meshes_paths = [f for f in os.listdir(romp_meshes_dir) if f.endswith('.obj')]
    romp_meshes_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    print("length of meshes_paths:", romp_meshes_paths)

    height,width,channels = get_original_image_size(index)
    R = Renderer((1024,1024))

    out_path = os.path.join(dir, 'render-results', 'romp', index)
    os.makedirs(out_path, exist_ok=True)

    count = 0 

    # Loop over the meshes and render them
    for idx in range(0, len(romp_meshes_paths)-1):

        #print('ROMP:', idx, end='\r')
        
        # only look at the first estimated body TODO; render all 
        splits = romp_meshes_paths[idx].split("_")
        body_id = splits[1].split(".")[0]
        print("splits:", splits, "BODY_ID:", body_id)
        if body_id == "1":
            print("True")
            continue

        # background white image
        white_image = np.ones((1024,1024,3))*255

        # load mesh as trimesh object
        mesh = trimesh.load(os.path.join(romp_meshes_dir,romp_meshes_paths[idx]), process=False)

        # load estimated camera info
        cam = [1,1,0,0]

        # render mesh on top of the white image
        render_result = R.render(white_image, mesh.vertices, mesh.faces, ht=-0.5, weak_cam=cam, flip=True)

        render_result = R.render(render_result, mesh.vertices, mesh.faces, romp_scaler=2, angle=90, axis=[0,1,0], ht=1,  weak_cam=cam, flip=True)

        # remove white borders and resize back to original image shape
        x = height / (width/1024)
        x = int(x)
        white_border_size = (1024 - x) / 2
        white_border_size = int(white_border_size)
        render_result = render_result[white_border_size:white_border_size+x,:,:]
        render_result = cv2.resize(render_result, (width,height), interpolation = cv2.INTER_LINEAR)
        cv2.imwrite( os.path.join(out_path, '{:06d}.png'.format(count)), render_result)
        count += 1

    images2video(out_path)

def render_expose(index, dir):
    #----------------------------------------------------------
    # LOAD EXPOSE ESTIMATIONS # these meshes have 10745
    expose_meshes_dir = os.path.join(dir, 'RESULTS2/expose/{}/meshes/'.format(index))
    expose_cams_dir = os.path.join(dir, 'RESULTS2/expose/{}/cams/'.format(index))

    expose_meshes_paths = [f for f in os.listdir(expose_meshes_dir) if f.endswith('.obj')]
    expose_meshes_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    expose_cams_paths = [f for f in os.listdir(expose_cams_dir) if f.endswith('.npy')]
    expose_cams_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    print("length of meshes_paths:", expose_meshes_paths)

    out_path = os.path.join(dir, 'render-results', 'expose', index)
    os.makedirs(out_path, exist_ok=True)
    
    height,width,channels = get_original_image_size(index)
    R = Renderer((width,height))

    # 
    previous_frame_id = None
    im_count = 0

    # Loop over the meshes and render them
    for idx in range(0, len(expose_meshes_paths)-1):

        #print('ExPose:', idx, end='\r')
        
        # Get the frame_id (because we have multiple estimations per frame)
        print(expose_meshes_paths[idx])
        splits = expose_meshes_paths[idx].split(".")
        frame_id = splits[0].strip("0")
        
        print("splits:", splits, "frame_id:", frame_id)
        
        # Only render the first estimation per frame (meaning we will ignore the others TODO: render all estimations)
        if frame_id == previous_frame_id:
            print("True")
            previous_frame_id = frame_id
            continue
        previous_frame_id = frame_id

        # background white image
        white_image = np.ones((height, width,3))*255

        # load mesh as trimesh object
        mesh = trimesh.load(os.path.join(expose_meshes_dir,expose_meshes_paths[idx]), process=False)

        # load estimated camera info
        cam = np.load(os.path.join(expose_cams_dir,expose_cams_paths[idx]), allow_pickle=True)

        # render mesh on top of the white image
        render_result = R.render(white_image, mesh.vertices, mesh.faces, ht_cam=2, cam=cam, flip=True)
        
        # render side mesh on top of previous result 
        render_result = R.render(render_result, mesh.vertices, mesh.faces, angle=270, axis=[0,1,0], ht_cam=0.75, cam=cam, flip=True)

        cv2.imwrite( os.path.join(out_path, '{:06d}.png'.format(im_count)), render_result)
        im_count += 1

    images2video(out_path)

def render_vibe(index, dir):
    out_path = os.path.join(dir, 'render-results', 'vibe', index)
    os.makedirs(out_path, exist_ok=True)
    
    height,width,channels = get_original_image_size(index)
    R = Renderer((width,height))

    # ----------------------------------------------------------
    # LOAD VIBE ESTIMATIONS # these meshes have 6890 vertices
    vibe_meshes_dir = os.path.join(dir, 'RESULTS2/vibe/{}/meshes/000000'.format(index))
    vibe_meshes_paths = [f for f in os.listdir(vibe_meshes_dir) if f.endswith('.obj')]
    vibe_intrin_paths = [f for f in os.listdir(vibe_meshes_dir) if f.endswith('.npy')]
    vibe_meshes_paths.sort(key=lambda f: int(re.sub('\D', '', f)))
    vibe_intrin_paths.sort(key=lambda f: int(re.sub('\D', '', f)))

    # Loop over the meshes and render them
    for idx in range(0, len(vibe_meshes_paths)-1):

        print('VIBE:', idx, end='\r')
        
        # background white image
        white_image = np.ones((height, width,3))*255

        # load mesh as trimesh object
        mesh = trimesh.load(os.path.join(vibe_meshes_dir,vibe_meshes_paths[idx]), process=False)

        # load estimated camera info
        cam_intrin = np.load(os.path.join(vibe_meshes_dir,vibe_intrin_paths[idx]))

        # render mesh on top of the white image
        render_result = R.render(white_image, mesh.vertices, mesh.faces, ht=-1, weak_cam=cam_intrin)
        
        #mean_head, mean_hips = get_all_head_vertex_locations(mesh.vertices)

        # render side mesh on top of previous result 
        render_result = R.render(render_result, mesh.vertices, mesh.faces, angle=270, axis=[0,1,0], ht=1, weak_cam=cam_intrin)

        cv2.imwrite( os.path.join(out_path, '{:06d}.png'.format(idx)), render_result)

    images2video(out_path)

ballet = False
if ballet:
    direc = '/media/weber/Ubuntu2/ubuntu2/Human_Pose/QMUL-data/Evaluation-clips/Clips/Fitness/ballet'
else:
    direc = '/mnt/c7dd8318-a1d3-4622-a5fb-3fc2d8819579/CORSMAL/QMUL_DANCE_DATA/squats/50-clips'

for index in range(00,50):
    index = "{:02d}".format(index)
    print("INDEX:", index)

    render_decomr(index, direc)
    render_romp(index, direc)
    render_expose(index, direc)
    render_vibe(index, direc)

print("Done!")

