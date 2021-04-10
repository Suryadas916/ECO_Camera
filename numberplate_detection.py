import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

camera = cv2.VideoCapture(1)
nPlateCascade = cv2.CascadeClassifier("cascade classifiers/haarcascade_russian_plate_number.xml")
input_image = './num_plate.jpeg'


def capture_image(src=None):
    """
    Capture input image
    :param src:
    :return:
    """
    if src:
        img = cv2.imread(src)
    else:
        _, img = camera.read()
    return img


def detect_numberplate(image):
    numberplates = nPlateCascade.detectMultiScale(image, 1.1, 5)

    return numberplates


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
            cv2.putText(image, reg_number, (x, y - 5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
    cv2.imshow('image', image)


def get_vehicle_number(image):
    """
    Use OCR to find text from numberplate image
    :param image:
    :return:
    """
    number = pytesseract.image_to_string(image)
    return number

# while True:
#     # image = capture_image()
#     image = capture_image(src)
#     numberplates = detect_numberplate(image)
#     plotbb(image, numberplates)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
