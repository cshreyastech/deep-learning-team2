import cv2
import deep_utils
import numpy as np
import pickle
import os

if __name__ == '__main__':
    filename = "tsane.jpg"

    model = deep_utils.load_model("model.yaml", "weights.h5")
    labels = pickle.load(open("y_names.p", "rb"))

    start_point = None
    end_point = None
    cropping = False

    def shape_selection(event, x, y, flags, param):
        # grab references to the global variables
        global start_point, end_point, cropping

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        center_point = (x, y)
        if event == cv2.EVENT_LBUTTONDBLCLK:
            start_point = (center_point[0] - 112, center_point[1] - 112)
            end_point = (center_point[0] + 112, center_point[1] + 112)
            # draw a rectangle around the region of interest
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("image", image)

    image = cv2.imread(filename)
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", shape_selection)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if start_point is not None and end_point is not None:
        crop_img = clone[start_point[1]:end_point[1], start_point[0]:end_point[0]]

        # Normalize the input image
        crop_img = np.divide(crop_img, 255)

        # Predict who this is using the saved trained model
        output = model.predict(np.array([crop_img]), steps=1)

        one_hot = np.round(output)
        index = np.where(one_hot == 1)
        index = index[0][0]

        print(labels[index])
        cv2.waitKey(0)

    # close all open windows
    cv2.destroyAllWindows()
