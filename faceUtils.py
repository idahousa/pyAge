# Reference link: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
import numpy as np
import dlib
from skimage import  transform
from skimage import io
# ==================================================================================================#
# Date-Of-Code: 2018-03-20
# Author: datnt
# Descriptions
# Convert from shape structure of dlib library to array of landmark points
# ==================================================================================================#
def shape_to_np(shape, dtype ="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for index in range(0, 68):
        coords[index] = (shape.part(index).x, shape.part(index).y)
    return coords
# ==================================================================================================#
# Date-Of-Code: 2018-03-20
# Author: datnt
# Descriptions
# Detect the landmark points of faces in a given face image
# ==================================================================================================#
def face_landmark_detection(image, face_detector, shape_predictor, up_levels=2):
    face_rects = face_detector(image, up_levels)
    num_faces = len(face_rects)
    face_shapes = []
    for (i, frect) in enumerate(face_rects):
        fshape = shape_predictor(image, frect)
        fshape = shape_to_np(fshape)
        face_shapes.append(fshape)
    return face_shapes
# ==================================================================================================#
# Date-Of-Code: 2018-03-20
# Author: datnt
# Descriptions
# In-plane rotation compensation of face and extract face region images
# ==================================================================================================#
def extract_face_region(image, face_shapes):
    num_faces = len(face_shapes)
    if num_faces < 1:
        return False
    faces = []
    for i in range(num_faces):
        current_face = extract_current_face(image, face_shapes[i])
        faces.append(current_face)
    return faces

def extract_current_face(image, face_shape):
    #Calculate the center of face region#
    center = np.array((0,0), dtype=np.int)
    left_eye = np.array((0,0), dtype=np.int)
    right_eye = np.array((0,0), dtype=np.int)
    for i in range(68):
        #image[face_shape[i][1]-2:face_shape[i][1]+2,face_shape[i][0]-2:face_shape[i][0]+2,:]=255
        if(i>=31 and i<=35):
            #Center of nose
            center[0] += face_shape[i][0]  # column index
            center[1] += face_shape[i][1]  # row index
        if (i>=36 and i<=41):
            left_eye[0] += face_shape[i][0]#column index
            left_eye[1] += face_shape[i][1]#row index
        if (i>=42 and i<=47):
            right_eye[0] += face_shape[i][0]#column index
            right_eye[1] += face_shape[i][1]#row index
    center[0] /= 5
    center[1] /= 5
    left_eye[0] /=6
    left_eye[1] /=6
    right_eye[0] /=6
    right_eye[1] /=6
    image[(center[1]-2):(center[1]+2), (center[0]-2):(center[0]+2),:]=255
    #image[(left_eye[1]-2):(left_eye[1]+2), (left_eye[0]-2):(left_eye[0]+2),:]=255
    #image[(right_eye[1] - 2):(right_eye[1] + 2), (right_eye[0] - 2):(right_eye[0] + 2), :] = 255
    #Rotate the original image and corresponding points
    dif_x = right_eye[0] - left_eye[0]
    dif_y = right_eye[1] - left_eye[1]
    tan_angle = dif_y/dif_x
    rot_angle = (180*np.arctan(tan_angle))/np.pi
    print(rot_angle)
    #Extract the face region and resizing
    rot_image, tform = rotate_image(image, rot_angle,resize=True)
    for i in range(68):
        face_shape[i] = cal_coordinate_of_pixel_in_rotated_image(tform=tform, pixel_coordinate = [ face_shape[i][1],  face_shape[i][0]] )
        rot_image[face_shape[i][1]-2:face_shape[i][1]+2,face_shape[i][0]-2:face_shape[i][0]+2,:]=1
    #io.imshow(image)
    #print(face_shape)
    #io.imshow(rot_image)
    #io.show()
    roi_x_low = face_shape[0][0]
    roi_y_low = face_shape[24][1]
    roi_x_high = face_shape[16][0]
    roi_y_high = face_shape[9][1]
    face = rot_image[roi_y_low:roi_y_high,roi_x_low:roi_x_high,:]
    return face
#==================================================================================================#
# ==================================================================================================#
# Date-of-code: 2018-03-20
# Author: datnt
# Description:
# Rotate an image arround an center. This function is same as the skimage.transform.rotate() function
# However, this function additionally return the transformation function that can be futher used
# to calculate the new locations of some landmark points
# Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/transform/_warps.py#L285
# Please import the transform module by using "from skimage import transform" statement
# ==================================================================================================#
def rotate_image(image, angle, resize=False, center=None, order=1, mode='constant', cval=0, clip=True, preserve_range=False):
    rows, cols = image.shape[0], image.shape[1]
    # rotation around center
    if center is None:
        center = np.array((cols, rows)) / 2. - 0.5
    else:
        center = np.asarray(center)
    tform1 = transform.SimilarityTransform(translation=center)
    tform2 = transform.SimilarityTransform(rotation=np.deg2rad(angle))
    tform3 = transform.SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1
    output_shape = None
    if resize:
        # determine shape of output image
        corners = np.array([[0, 0],
                            [0, rows - 1],
                            [cols - 1, rows - 1],
                            [cols - 1, 0]])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.ceil((out_rows, out_cols))
        # fit output image in new shape
        translation = (minc, minr)
        tform4 = transform.SimilarityTransform(translation=translation)
        tform = tform4 + tform
    r_image = transform.warp(image, tform, output_shape=output_shape, order=order, mode=mode, cval=cval, clip=clip, preserve_range=preserve_range)
    return r_image, tform
#==================================================================================================#
# Date-of-code: 2018-03-20
# Author: datnt
# Description:
# This function is used after the use of rotate_image() function above to calculate the new coordinate of a given pixel
# of original images in the rotated image.
# A given pixel is in format of [row, column].
"""
Example:
>> image = data.camera()
>> image[ 210:220, 400:410] = 255
>> ri_image, tform = rotate_image(image,-10,resize=True)
>> new_coordinate = cal_coordinate_of_pixel_in_rotated_image(tform=tform,pixel_coordinate = [215, 405])
>> print(new_coordinate)
>> io.imshow(ri_image)
>> io.show()
"""
def cal_coordinate_of_pixel_in_rotated_image(tform, pixel_coordinate):
    point_array = np.array([[pixel_coordinate[1], pixel_coordinate[0]]])
    return tform.inverse(point_array)
#==================================================================================================#
