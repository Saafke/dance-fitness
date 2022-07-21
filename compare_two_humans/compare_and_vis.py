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
# The code herein makes us of the SMPL-Body, 
# which is licensed under the Creative Commons Attribution 4.0 International License
# License link: https://smpl.is.tue.mpg.de/bodylicense.html

"""
Compares two SMPL meshes.
Computes their discrepancies and visualises them.
"""
import numpy as np
import trimesh
import math
import cv2
import json
import pyrender
from pyrender.constants import RenderFlags

# LIMBS, i.e. connections between joint indices
connections = [
	# head
	[17,15],
	[15,0],
	[0,16],
	[16,18],
	# neck
	[0,1],
	# right arm 
	[1,2],
	[2,3],
	[3,4],
	# left arm 
	[1,5],
	[5,6],
	[6,7],
	# right leg
	[1,8],
	[8,9],
	[10,9],
	[11,10],
	[11,24],
	[11,22],
	[22,23],
	# left leg
	[8,12],
	[12,13],
	[13,14],
	[14,21],
	[14,19],
	[19,20]
 ]

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

def change_order(score):
	"""
	Changing indices.
	"""
	ordered_score = []
	ordered_score.append(2)
	ordered_score.append(score[13])
	ordered_score.append(score[9])
	ordered_score.append(score[20])
	ordered_score.append(score[23])
	ordered_score.append(score[22])
	ordered_score.append(score[11]) # spine
	ordered_score.append(score[11]) # spine
	ordered_score.append(score[8])
	ordered_score.append(score[5])
	ordered_score.append(score[16])
	ordered_score.append(score[2]) # TODO heads are more numbers
	ordered_score.append(score[6])
	ordered_score.append(2)
	ordered_score.append(score[14])
	ordered_score.append(2)
	ordered_score.append(score[10])
	ordered_score.append(score[7])
	ordered_score.append(score[4])
	ordered_score.append(score[17])
	ordered_score.append(score[11]) # spine
	ordered_score.append(score[19])
	ordered_score.append(2)
	ordered_score.append(score[18]) # TODO hips are two numbers

	return ordered_score

def idx2color(vert_idx, score, labels):
	"""
	Return red if score == 0 else green
	
	Semantic body parts : Connection : Connection Index
	
	rightHand : -
	rightUpLeg : 9-10 : 13
	leftArm : 5-6 : 9
	leftLeg : 13-14 : 20
	leftToeBase : 19-20 : 23
	leftFoot : 14-19 : 22
	spine1 : 1-8 : 11
	spine2 : 1-8 : 11
	leftShoulder : 1-5 : 8
	rightShoulder : 1-2 : 5
	rightFoot : 11-22 : 16
	head : 0-16 : 2 TODO fix this
	rightArm : 2-3 : 6
	leftHandIndex1 : -
	rightLeg : 10-11 : 14
	rightHandIndex1 : -
	leftForeArm : 6-7 : 10
	rightForeArm : 3-4 : 7
	neck : 0-1 : 4
	rightToeBase : 22-23 : 17
	spine : 1-8 : 11
	leftUpLeg : 12-13 : 19
	leftHand : -
	hips : 8-12 / 8-9 : 18/12
	
	"""
	score = change_order(score)

	semantic_body_parts = labels.keys()
	
	for idx, part in enumerate(semantic_body_parts):
		# Check if current vertex belongs to this body part
		if vert_idx in labels[part]:

			# Check the score of this body part. BGR FORMAT
			if score[idx] == 0:
				return [0.0, 0, 1.0] # incorrect, red (colourblind version: orange (255,165,0)) 
			elif score[idx] == 1:
				return [0, 1.0, 0.0] # correct, green (colourblind version: blue)
			else:
				return [0.0,0.0,0.0] # no matching, black
	return [0.0,0.0,0.0] # black

def render(img, score, verts, faces, cam, angle=None, axis=None, color=[1.0, 1.0, 1.0], translate_left=False, wireframe=True):

	# Set the Trimesh Mesh
	mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

	# Color the parts of the mesh based on the score!
	if score is not None:
		
		# Online
		f = open('./data/smpl_vert_segmentation.json',)
		labels = json.load(f)

		# Loop over the vertices and colour them
		vert_colors = []
		for idx, vert in enumerate(verts):
			
			# Check to which semantic body part this label belongs to
			vert_color = idx2color(idx, score, labels)

			# if idx in labels["spine1"]:
			#     vert_colors.append([0.0,0.0,1.0])
			# else:
			#     vert_colors.append([0.0,1.0,0.0])
			vert_colors.append(vert_color)

		mesh.visual.vertex_colors = vert_colors

		color_0, texcoord_0, primitive_material = pyrender.Mesh._get_trimesh_props(mesh)
	############

	# transform
	Rx = trimesh.transformations.rotation_matrix(math.radians(0), [1, 0, 0])
	mesh.apply_transform(Rx)
	if angle and axis:
		R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
		mesh.apply_transform(R)

	# Load weak-perspective camera data
	sx, sy, tx, ty = cam

	# Render mesh a bit to the left
	if translate_left:
		tx-=1
	else:
		tx+=1
		#ty+=0.1
	camera = WeakPerspectiveCamera(
		scale=[sx, sy],
		translation=[tx, ty],
		zfar=1000.
	)

	# material = pyrender.MetallicRoughnessMaterial(
	#     metallicFactor=0.0,
	#     alphaMode='OPAQUE',
	#     baseColorFactor=(color[0], color[1], color[2], 1.0)
	# )

	if score is not None:
		mesh = pyrender.Mesh.from_trimesh(mesh)
	else:
		material = pyrender.MetallicRoughnessMaterial(
			metallicFactor=0.0,
			alphaMode='MASK',
			baseColorFactor=(color[0], color[1], color[2], 0.2) # 0.5
		)
		mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

	
	mesh_node = scene.add(mesh, 'mesh')

	camera_pose = np.eye(4)
	cam_node = scene.add(camera, pose=camera_pose)

	# Rendering options
	if wireframe:
		render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
	elif flat:
		render_flags = RenderFlags.RGBA | RenderFlags.FLAT
	else:
		render_flags = RenderFlags.RGBA

	# Render
	rgb, _ = renderer.render(scene, flags=render_flags)

	# Mask image accordingly	
	valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
	output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
	image = output_img.astype(np.uint8)

	scene.remove_node(mesh_node)
	scene.remove_node(cam_node)

	cv2.imwrite('./output.png', image)
	return image

def compute_score(skel_1, skel_2, angle_margin=5, radius=.2, num_seg=10, scale=.1):
	"""
	Computes the discrepancies between skeleton 1 and 2.
	"""

	# HELPER FUNCTIONS
	def extend_line(start_pt, end_pt, scale):
		v = end_pt - start_pt
		normalized_v = v / np.sqrt(np.sum(v**2))

		start_pt = start_pt - scale * normalized_v
		end_pt = end_pt + scale * normalized_v

		return start_pt, end_pt
	
	def segment_line(start_pt, end_pt, num_seg):
		seg = []
		for k in range(num_seg + 1):
			t = float(k) / num_seg
			C_k = start_pt * (1-t) + end_pt * t
			seg += [list(C_k)]
		return seg
	
	def point_in_cylinder(pt1, pt2, r, q):
		'''
		Determine if a query 3D point q is inside a Cylinder.
		The Cylinder is defined by pt1, pt2 and a radius r
		'''
		dx = pt2[0] - pt1[0]
		dy = pt2[1] - pt1[1]
		dz = pt2[2] - pt1[2]

		pdx = q[0] - pt1[0]
		pdy = q[1] - pt1[1]
		pdz = q[2] - pt1[2]

		dot = pdx * dx + pdy * dy + pdz * dz

		length = pt1-pt2
		lengthsq = length.dot(length)

		if dot < 0 or dot > lengthsq:
			return -1
		else:
			dsq = (pdx*pdx + pdy*pdy + pdz*pdz) - dot*dot/lengthsq
			if dsq > (r ** 2):
				return -1
			else:
				return 1

	def unit_vector(vector):
		""" Returns the unit vector of the vector.  """
		return vector / np.linalg.norm(vector)

	def angle_between(v1, v2):
		""" Returns the angle in radians between vectors 'v1' and 'v2'
		"""
		v1_u = unit_vector(v1)
		v2_u = unit_vector(v2)
		return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
	
	
	score = []
	
	# Loop over the limbs (i.e. edges connected to joints)
	for connection in connections:

		# Get joint indices
		start_idx = connection[0]
		end_idx = connection[1]

		# Get 3D points of these joint
		pt1 = skel_1[start_idx,:]
		pt2 = skel_1[end_idx,:]
		pt1_, pt2_ = extend_line(pt1, pt2, scale) # extend the teacher's line
		
		# Get 3D points these joint
		ps1 = skel_2[start_idx,:]
		ps2 = skel_2[end_idx,:]
		line_seg = segment_line(ps1, ps2, num_seg) # divide the student line into equal segement

		# Compute discrepancies by looking at the angles
		min_seg, max_seg = -1, -1
		for k, seg in enumerate(line_seg):
			in_ = point_in_cylinder(pt1_, pt2_, radius, seg)
			if in_ == 1:
				if min_seg == -1:
					min_seg = k
				if max_seg <= k:
					max_seg = k
		if min_seg != 1 and max_seg != -1:
			ang_t = np.degrees(angle_between(pt1, pt2))
			ang_s = np.degrees(angle_between(ps1, ps2))

			if abs(ang_t - ang_s) <= angle_margin:
				perc = (max_seg - min_seg) / float(num_seg)
				#score += [perc]
				score += [1.]
			else:
				score += [0]
		else:
			score += [0]
		#scores.append(score)

	return score

### Toy-example, using SMPL body meshes
### You can also make your own meshes/skeletons/camera parameters, by using e.g. VIBE

# Get meshes
SMPL_mesh_1 = "./data/mesh1.obj"
SMPL_mesh_2 = "./data/mesh2.obj"
# load mesh via trimesh
trimesh_1 = trimesh.load(SMPL_mesh_1, process=False)
trimesh_2 = trimesh.load(SMPL_mesh_2, process=False)

# Load the corresponding 3D skeleton data - you could also extract it from the mesh [check VIBE code]
skel_1 = np.load("./data/skel1.npy")
skel_2 = np.load("./data/skel2.npy")

# Get camera INTRINSIC parameters
cam_1 = np.load("./data/cam1.npy")
cam_2 = np.load("./data/cam2.npy")

# Compute scores
score = compute_score(skel_1, skel_2, angle_margin=5, radius=.2, num_seg=10, scale=.1)


# Init white image for rendering background
width = 1920
height = 1080
image = np.ones((height,width,3))*255

# Set the 3D scene in pyrender - this is for rendering the 3D mesh onto the image plane
scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))
light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
light_pose = np.eye(4)
light_pose[:3, 3] = [0, -1, 1]
scene.add(light, pose=light_pose)
light_pose[:3, 3] = [0, 1, 1]
scene.add(light, pose=light_pose)
light_pose[:3, 3] = [1, 1, 2]
scene.add(light, pose=light_pose)
renderer = pyrender.OffscreenRenderer(
	viewport_width=width,
	viewport_height=height,
	point_size=1.0
)

## Rendering
angle,axis=None,None
# paint first mesh - REFERENCE
image = render(image, None, trimesh_1.vertices, trimesh_1.faces, cam_1, angle=angle, axis=axis, translate_left=True)
# paint second mesh - COLORED MESH
image = render(image, score, trimesh_2.vertices, trimesh_2.faces, cam_2, angle=angle, axis=axis, translate_left=False)

print("\nRan successfully. See ./output.png for your result!\n")