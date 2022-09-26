"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under the Apache License, Version 2.0
"""

import cv2
import numpy as np
import os
import math

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from model import build_EfficientPose
from utils import preprocess_image
from utils.visualization import draw_detections

import rospy
# import tf2_ros
# import tf_conversions
from scipy.spatial.transform import Rotation as R

import geometry_msgs.msg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import unittest.mock as mock

class EfficientPose(object):
    def __init__(self):

        self.frame = None
        self.bridge = CvBridge()

        self.params = mock.Mock()

        self.get_params()

        self.model, self.image_size = build_model_and_load_weights(self.params)

        self.publisher = rospy.Publisher("efficient_pose/tf", geometry_msgs.msg.TransformStamped)

        rospy.Subscriber(self.params.input, Image, self.image_callback, queue_size=1000)


    def spin(self):
        rospy.wait_for_message(self.params.input, Image)

        img_copy = self.frame.copy()

        input_list, scale = preprocess(img_copy, self.image_size, self.params.camera_matrix, self.params.translation_scale_norm)

        boxes, scores, labels, rotations, translations = self.model.predict_on_batch(input_list)


        boxes, scores, labels, rotations, translations = postprocess(boxes, scores, labels, rotations, translations, scale, self.params.score_threshold)


        if self.params.visualize:
            draw_detections(self.frame,
                            boxes,
                            scores,
                            labels,
                            rotations,
                            translations,
                            class_to_bbox_3D=self.params.box_3d,
                            camera_matrix=self.params.camera_matrix,
                            label_to_name=self.params.class_name,
                            draw_bbox_2d=self.params.draw_bbox_2d,
                            draw_name=self.params.draw_name)

            # display image with predictions
            cv2.imshow('image with predictions', self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()

        print(0)
        if translations is not None:
            for trans, rot in zip(translations, rotations):
                t = geometry_msgs.msg.TransformStamped()
                # t.header = rospy.Time.now()
                # t.child_frame_id = "tf"
                t.transform.translation.x = trans[0]
                t.transform.translation.y = trans[1]
                t.transform.translation.z = trans[2]
                q = R.from_euler('zyx',rot).as_quat()
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]

                self.publisher.publish(t)




    def image_callback(self, ros_image):
        """Callback when a new image arrives, transforms it in cv2 image and set self.new_image to True"""
        frame = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")

        self.frame = np.array(frame, dtype=np.uint8)

    def get_params(self):
        self.params.path_to_weights = rospy.get_param("efficient_pose/path_to_weights")
        self.params.score_threshold = rospy.get_param("efficient_pose/score_threshold")
        self.params.translation_scale_norm = rospy.get_param("efficient_pose/translation_scale_norm")
        self.params.draw_bbox_2d = rospy.get_param("efficient_pose/draw_bbox_2d")
        self.params.draw_name = rospy.get_param("efficient_pose/draw_name")
        self.params.input = rospy.get_param("efficient_pose/input")
        self.params.visualize = rospy.get_param("efficient_pose/visualize")
        camera_matrix = rospy.get_param("efficient_pose/camera_matrix")
        self.params.camera_matrix = np.array(camera_matrix).reshape(3,3)
        self.params.phi = rospy.get_param("efficient_pose/phi")
        class_name = rospy.get_param("efficient_pose/class_names")
        self.params.class_name = dict()
        for i, name in enumerate(class_name):
            self.params.class_name[i] = name
        class_3dbox = rospy.get_param("efficient_pose/class_3dbox")
        self.params.box_3d = dict()
        for i, box in enumerate(class_3dbox):
            self.params.box_3d[i] = convert_bbox_3d(box)

def convert_bbox_3d(model_dict):
    """
    Converts the 3D model cuboids from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
    Args:
        model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
    Returns:
        bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.

    """
    # get infos from model dict
    min_point_x = model_dict[1]
    min_point_y = model_dict[2]
    min_point_z = model_dict[3]

    size_x = model_dict[4]
    size_y = model_dict[5]
    size_z = model_dict[6]

    bbox = np.zeros(shape=(8, 3))
    # lower level
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    # upper level
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])

    return bbox
    
def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)





def build_model_and_load_weights(params):
    """
    Builds an EfficientPose model and init it with a given weight file
    Args:
        params: all parameters
    Returns:
        efficientpose_prediction: The EfficientPose model
        image_size: Integer image size used as the EfficientPose input resolution for the given phi

    """
    print("\nBuilding model...\n")
    _, efficientpose_prediction, _ = build_EfficientPose(params.phi,
                                                         num_classes = 1,
                                                         num_anchors = 9,
                                                         freeze_bn = True,
                                                         score_threshold = params.score_threshold,
                                                         num_rotation_parameters = 3,
                                                         print_architecture = False)
    
    print("\nDone!\n\nLoading weights...")
    efficientpose_prediction.load_weights(params.path_to_weights, by_name=True)
    print("Done!")
    
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[params.phi]
    
    return efficientpose_prediction, image_size


def preprocess(image, image_size, camera_matrix, translation_scale_norm):
    """
    Preprocesses the inputs for EfficientPose
    Args:
        image: The image to predict
        image_size: Input resolution for EfficientPose
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_list: List containing the preprocessed inputs for EfficientPose
        scale: The scale factor of the resized input image and the original image

    """
    image = image[:, :, ::-1]
    image, scale = preprocess_image(image, image_size)
    camera_input = get_camera_parameter_input(camera_matrix, scale, translation_scale_norm)
    
    image_batch = np.expand_dims(image, axis=0)
    camera_batch = np.expand_dims(camera_input, axis=0)
    input_list = [image_batch, camera_batch]
    
    return input_list, scale


def get_camera_parameter_input(camera_matrix, image_scale, translation_scale_norm):
    """
    Return the input vector for the camera parameter layer
    Args:
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        image_scale: The scale factor of the resized input image and the original image
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_vector: numpy array [fx, fy, px, py, translation_scale_norm, image_scale]

    """
    #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
    input_vector = np.zeros((6,), dtype = np.float32)
    
    input_vector[0] = camera_matrix[0, 0]
    input_vector[1] = camera_matrix[1, 1]
    input_vector[2] = camera_matrix[0, 2]
    input_vector[3] = camera_matrix[1, 2]
    input_vector[4] = translation_scale_norm
    input_vector[5] = image_scale
    
    return input_vector


def postprocess(boxes, scores, labels, rotations, translations, scale, score_threshold):
    """
    Filter out detections with low confidence scores and rescale the outputs of EfficientPose
    Args:
        boxes: numpy array [batch_size = 1, max_detections, 4] containing the 2D bounding boxes
        scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
        labels: numpy array [batch_size = 1, max_detections] containing class label
        rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
        scale: The scale factor of the resized input image and the original image
        score_threshold: Minimum score threshold at which a prediction is not filtered out
    Returns:
        boxes: numpy array [num_valid_detections, 4] containing the 2D bounding boxes
        scores: numpy array [num_valid_detections] containing the confidence scores
        labels: numpy array [num_valid_detections] containing class label
        rotations: numpy array [num_valid_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [num_valid_detections, 3] containing the translation vectors

    """
    boxes, scores, labels, rotations, translations = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
    # correct boxes for image scale
    boxes /= scale
    #rescale rotations
    rotations *= math.pi
    #filter out detections with low scores
    indices = np.where(scores[:] > score_threshold)
    # select detections
    scores = scores[indices]
    boxes = boxes[indices]
    rotations = rotations[indices]
    translations = translations[indices]
    labels = labels[indices]
    
    return boxes, scores, labels, rotations, translations


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()

    rospy.init_node('efficient_pose_node', anonymous=True)
    efficient_node = EfficientPose()

    while not rospy.is_shutdown():
        efficient_node.spin()

