# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
from imutils import face_utils
import dlib
import time
import cv2
import os
from face_mapping_analysis import eye_closed_detection, no_face_detection
from object_detection_analysis import cell_phone_detection

# Frame rate of input video
FRAME_RATE = 30

# Eye aspect ratio threshold
EYE_AR_THRESHOLD = 0.3

# Threshold for seconds without face
EYES_CLOSED_THRESHOLD = 2

# Threshold for seconds without face
NO_FACE_THRESHOLD = 2

# Threshold for seconds with phone
PHONE_THRESHOLD = 2

# Convert threshold values from seconds to frames
EYES_CLOSED_THRESHOLD *= FRAME_RATE
NO_FACE_THRESHOLD *= FRAME_RATE
PHONE_THRESHOLD *= FRAME_RATE

# construct the argument parse and parse the arguments
yolo_path = 'yolo-coco'

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video")
ap.add_argument("-o", "--output", required=True,
    help="path to output video")
ap.add_argument("-y", "--yolo", default=yolo_path,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels the YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# count current frame for feedback in terminal
current = 0

# count frames with no face detected
no_face_count = 0

# count frames with closed eyes
eyes_closed_count = 0

# count frames with phone
phone_count = 0

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # find computational time per frame
    start = time.time()
    layerOutputs = net.forward(ln)

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions, detected probability > minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

                # DLIB head pose estimation - convert input image to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # detect faces in the grayscale image
                rects = detector(gray, 0)

                # if no face is detected, count for how long
                no_face_count = no_face_detection(rects, no_face_count)

                # if no face detected, reset eyes closed counter
                if no_face_count > 0:
                    eyes_closed_count = 0

                if no_face_count > NO_FACE_THRESHOLD:
                    # draw inattention detection box
                    cv2.putText(frame, "INATTENTION DETECTED: driver is looking in the wrong direction", (10, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # loop over the face detections
                for (i, rect) in enumerate(rects):
                    # determine the facial landmarks for the face region
                    shape = predictor(gray, rect)

                    # convert the facial landmark (x, y)-coordinates to a NumPy array
                    shape = face_utils.shape_to_np(shape)

                    # loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
                    for (x, y) in shape:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                    # count number of frames with eyes closed
                    eyes_closed_count = eye_closed_detection(shape, eyes_closed_count, EYE_AR_THRESHOLD)

                    # if the eyes were closed for more frames than threshold
                    if eyes_closed_count >= EYES_CLOSED_THRESHOLD:
                        # draw inattention detection box
                        cv2.putText(frame, "INATTENTINO DETECTED: the eyes of the driver is closed", (10, 500),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # Compute seconds from frame
    no_face_sec = no_face_count/FRAME_RATE
    eyes_closed_sec = eyes_closed_count/FRAME_RATE
    phone_sec = phone_count/FRAME_RATE

    # draw seconds with no face, 2 digits after comma
    cv2.putText(frame, "No face for: {} seconds".format("%.2f" % no_face_sec), (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # draw seconds with eyes closed, 2 digits after comma
    cv2.putText(frame, "Eyes closed for: {} seconds".format("%.2f" % eyes_closed_sec), (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # draw seconds of frames with eyes closed, 2 digits after comma
    cv2.putText(frame, "Phone used for: {} seconds".format("%.2f" % phone_sec), (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # find computational time per frame
    end = time.time()
    elap = (end - start)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            phone_count = cell_phone_detection(text, phone_count)

            if phone_count > PHONE_THRESHOLD:
                # draw inattention detection box
                cv2.putText(frame, "INATTENTINO DETECTED: phone used", (10, 600),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

        # some information on processing single frame
        if total > 0:
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish (in seconds): {:.4f}".format(elap * total))
            print("[INFO] estimated total time to finish (in minutes): {:.4f}".format((elap * total)/60))

    # write the output frame to disk
    writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
print("Finished")