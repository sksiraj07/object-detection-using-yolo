import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video or image")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
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

# Determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()

# Fix for different versions of OpenCV
try:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(image):
    (H, W) = image.shape[:2]

    # construct a blob from the input image and then perform a forward pass of the YOLO object detector
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
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

            # filter out weak predictions by ensuring the detected probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of the bounding box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

input_path = args["input"]

if not os.path.exists(input_path):
    print(f"[ERROR] Input file {input_path} does not exist.")
    exit()

if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    # process image
    print("[INFO] processing image...")
    image = cv2.imread(input_path)
    if image is None:
        print(f"[ERROR] Could not read image at {input_path}.")
        exit()
    result_image = detect_objects(image)
    cv2.imshow("Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    # process video
    print("[INFO] processing video...")
    vs = cv2.VideoCapture(input_path)
    if not vs.isOpened():
        print("[ERROR] Could not open video file.")
        exit()

    (W, H) = (None, None)

    # get the input video's frame rate
    fps = vs.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("[ERROR] Frames per second could not be determined.")
        exit()
    frame_delay = int(1000 / fps)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # if the frame dimensions are empty, grab them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # detect objects in the frame
        start = time.time()
        result_frame = detect_objects(frame)
        end = time.time()

        # calculate the processing time and adjust the wait time accordingly
        processing_time = (end - start) * 1000  # convert to milliseconds
        wait_time = max(int(frame_delay - processing_time), 1)

        # display the output frame
        cv2.imshow("Frame", result_frame)
        key = cv2.waitKey(wait_time) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # some information on processing single frame
        if total > 0:
            print("[INFO] single frame took {:.4f} seconds".format(end - start))
            print("[INFO] estimated total time to finish: {:.4f}".format((end - start) * total))

    # release the file pointers
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    vs.release()
