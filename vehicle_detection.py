import time
import cv2
import os
import numpy as np

confi = 0.5
threshold = 0.3

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join(['yolo-coco', "yolov3.weights"])
configPath = os.path.sep.join(['yolo-coco', "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# load our input image and grab its spatial dimensions
image = cv2.imread('./car.jpg')

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_vehicle(image):
    (H, W) = image.shape[:2]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    class_ids = [2, 3, 5, 7, ]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confi:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confi,
                            threshold)

    # ensure at least one detection exists
    b = []
    c = []
    id = []
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            if classIDs[i] in class_ids:
                # extract the bounding box coordinates
                b.append(boxes[i])
                id.append(classIDs[i])
                c.append(confidences[i])

                return b, c, id


def plotbb(image, detections):
    """
    Plot Bounding box
    :param image:
    :param detections:
    :return:
    """
    for d in detections:
        [x, y, w, h] = d
        area = w * h
        min_area = 200
        if area > min_area:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # roi = image[y:y + h, x:x + w]
            # reg_number = get_vehicle_number(roi)
            # cv2.putText(image, reg_number, (x, y - 5),
            #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    cv2.imshow('image', image)
    cv2.waitKey(0)


if __name__ == "__main__":
    detections = detect_vehicle(image)
    print(detections)
    boxes = detections[0]

    plotbb(image, boxes)
