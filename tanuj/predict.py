import cv2
import deep_utils
import numpy as np
import pickle

if __name__ == '__main__':
    img_list = np.array([cv2.imread("tsane.jpg")])

    # Predict who this is using the saved trained model
    model = deep_utils.load_model("facial_recognition_large_cnn_model.yaml", "facial_recognition_large_cnn_model.h5")
    output = model.predict(img_list, steps=1)

    # Map the one-hot output to the correct label
    labels = pickle.load(open("y_names.p", "rb"))
    index = np.where(output == 1.0)

    print(f"This is {labels[index]}")



