#Last login: Thu Jan 21 11:14:51 on ttys002
#(base) nithinvenkat@student-10-201-23-058 ~ % ipython
#Python 3.7.6 (default, Jan  8 2020, 13:42:34)
#Type 'copyright', 'credits' or 'license' for more information
#IPython 7.12.0 -- An enhanced Interactive Python. Type '?' for help.

# import the necessary packages
from collections import OrderedDict
#change
from scipy.spatial import distance as dist
import numpy as np
import cv2
import argparse
import dlib
import imutils

facial_features_cordinates = {}

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates
    
#    start of changes
def mouth_aspect_ratio(mouth):
    # compute the euclidean distances between the two sets of
    # vertical inner mouth landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[1], mouth[7])
    B = dist.euclidean(mouth[2], mouth[6])
    C = dist.euclidean(mouth[3], mouth[5])
 
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    D = dist.euclidean(mouth[0],mouth[4])
 
    # compute the mouth aspect ratio
    mouth_ratio = (A +B+C) / (3.0 * D)
 
    # return the eye aspect ratio
    return mouth_ratio
    
#    end of changes


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    print(facial_features_cordinates)
    return output

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
    shape = predictor(gray, rect)
    shape = shape_to_numpy_array(shape)
    # visualize all facial landmarks with a transparent overlay
    output = visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
#    start of change
    pt1=np.int0(shape[[61,60,67]].mean(axis=0))
    pt2=np.int0(shape[[63,64,65]].mean(axis=0))
    pt3=np.int0(shape[[63,64,55]].mean(axis=0))
    pt4=np.int0(shape[[61,60,59]].mean(axis=0))

    mouth_thres=0.17

    pts_inner_mouth=shape[[60,61,62,63,64,65,66,67]]
    #print(mouth_aspect_ratio(pts_inner_mouth))
    if(mouth_aspect_ratio(pts_inner_mouth)<mouth_thres or mouth_aspect_ratio(pts_inner_mouth)>0.5):
        print('----------------------------------------------------------------------')
        print('TEETH IS NOT VISIBLE')
        print('----------------------------------------------------------------------')
        break
print('----------------------------------------------------------------------')
print('TEETH IS VISIBLE')
print('----------------------------------------------------------------------')
#    print('TEETH IS VISIBLE')
        
#        end of change

