import cv2
import deep_utils
import numpy as np
import pickle
import os

if __name__ == '__main__':
    file_list = [f"test/{img}" for img in os.listdir("test")]
    img_list = np.array([np.divide(cv2.imread(file), 255) for file in file_list])

    # Predict who this is using the saved trained model
    model = deep_utils.load_model("facial_recognition_large_cnn_model.yaml", "facial_recognition_large_cnn_model.h5")
    output = model.predict(img_list, steps=1)

    # Map the one-hot output to the correct label
    labels = pickle.load(open("y_names.p", "rb"))

    for i in range(len(file_list)):
        filename = file_list[i]

        one_hot = np.round(output[i])
        index = np.where(one_hot == 1)
        index = index[0][0]

        print(f"{filename} is a picture of {labels[index]}")