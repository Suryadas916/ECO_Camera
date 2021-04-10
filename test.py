import cv2
from vehicle_detection import detect_vehicle
from numberplate_detection import detect_numberplate, get_vehicle_number

# load our input image and grab its spatial dimensions
image = cv2.imread('./car2.jpg')
try:
    vehicles = detect_vehicle(image)
    # Do numberplate detection and garbage detection if vehicles are detected
    # TODO Garbage detection
    # if garbage and vehicle in close proximity, do numberplate detection

    if len(vehicles) > 0:
        bounding_boxes = vehicles[0]
        for bb in bounding_boxes:
            [x, y, w, h] = bb

            area = w * h
            min_area = 200
            if area > min_area:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
                # Numberplate detection
                numberplates = detect_numberplate(image)
                if len(numberplates) > 0:
                    [xn, yn, wn, hn] = numberplates[0]
                    numberplate_roi = image[yn:yn + hn, xn:xn + wn]
                    reg_number = get_vehicle_number(numberplate_roi)
                    if reg_number:
                        cv2.putText(image, 'Vehicle Reg No: ' + reg_number, (x, y - 5),
                                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        cv2.imshow('output', image)
        cv2.waitKey(0)



    else:
        print('No vehicles detected')
except Exception as e:
    print(e)
