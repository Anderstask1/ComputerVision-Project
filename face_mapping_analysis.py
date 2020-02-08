from scipy.spatial import distance as dist
from imutils import face_utils
import cv2

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear

def eye_closed_detection(shape, counter, EYE_AR_THRESH):
    # initialize the frame counters and the total number of blinks
    TOTAL = 0

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    # extract the left and right eye coordinates
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    #  use the coordinates to compute the eye aspect ratio for both eyes
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0

    # check if ear is below threshold, if so increment the blink frame counter
    if ear < EYE_AR_THRESH:
        counter += 1
    # otherwise, the eye aspect ratio is not below the blink threshold
    else:
        # reset the eye frame counter
        counter = 0

    return counter

def no_face_detection(rects, counter):
    # if no face is detected, count for how long
    if len(rects) == 0:
        return counter + 1
    else:
        return 0