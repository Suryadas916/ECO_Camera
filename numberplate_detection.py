import cv2
import pytesseract

camera = cv2.VideoCapture(0)
nPlateCascade = cv2.CascadeClassifier("cascade classifiers/haarcascade_russian_plate_number.xml")
src = './num_plate.jpeg'


def capture_image(src=None):
    if src:
        img = cv2.imread(src)
    else:
        _, img = camera.read()
    return img


def detect_numberplate(image):
    numberPlates = nPlateCascade.detectMultiScale(image, 1.1, 5)
    return numberPlates


def plotbb(image, detections):
    """
    Plot Bounding box
    :param image:
    :param detections:
    :return:
    """
    for (x, y, w, h) in detections:
        area = w * h
        min_area = 200
        if area > min_area:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
            roi = image[y:y + h, x:x + w]
            reg_number = get_vehicle_number(roi)
            print(reg_number)


def get_vehicle_number(image):
    number = pytesseract.image_to_string(image)
    return number


while True:
    image = capture_image(src)
    numberplates = detect_numberplate(image)
    plotbb(image, numberplates)

    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
