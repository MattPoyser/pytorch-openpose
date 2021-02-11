import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

body_estimation = Body('/home2/lgfm95/openposehzzone/model/body_pose_model.pth')
hand_estimation = Hand('/home2/lgfm95/openposehzzone/model/hand_pose_model.pth')

def main(oriImg):
    shape0 = oriImg.shape
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    shape1 = canvas.shape
    canvas = util.draw_bodypose(canvas, candidate, subset)
    # detect hand
    hands_list = util.handDetect(candidate, subset, oriImg)

    all_hand_peaks = []
    shape2 = canvas.shape
    for x, y, w, is_left in hands_list:
        # cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        # cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)

    shape3 = canvas.shape
    canvas = util.draw_handpose(canvas, all_hand_peaks)
    shape4 = canvas.shape
    cv2.imwrite("test.png", canvas)

    raise AttributeError(shape0, shape1, shape2, shape3, shape4)
    return canvas[:, :, [2, 1, 0]]

if __name__ == '__main__':
    test_image = 'images/demo.jpg'
    oriImg = cv2.imread(test_image)  # B,G,R order
    canvas = main(oriImg)
    plt.imshow(canvas)
    plt.axis('off')
    plt.show()
