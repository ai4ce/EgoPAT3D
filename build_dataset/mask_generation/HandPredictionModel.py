import numpy as np
import cv2
import HandPrediction

def get_image(imageID, type):
 """
 Converts an image number into the file path where the image is located,
 opens the image, and returns the image as a numpy array.
 """
 img = cv2.imread('./' + HandPrediction.recording_name + '/color_frames/' + imageID)
 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 if type == "HSV":
  img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
 return img

def create_mask(n):
    test1 = get_image((n), "HSV")
    res = cv2.resize(test1, dsize=(240, 135), interpolation=cv2.INTER_CUBIC)
    generated_mask = np.zeros((135, 240, 3))
    for i in range(135):
        for x in range(240):
            if HandPrediction.clf_pure_predict.predict([res[i][x]])[0] == 0:
                generated_mask[i][x] = [0, 0, 0]
            else:
                generated_mask[i][x] = [255, 255, 255]
    kernel = np.ones((3, 3), np.uint8)
    generated_mask = cv2.morphologyEx(generated_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(str('./' + HandPrediction.recording_name + '/hand_frames/masks/mask' + n[-8:]), generated_mask)

