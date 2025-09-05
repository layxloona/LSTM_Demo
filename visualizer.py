import cv2
import numpy as np

def draw(test):
    back = cv2.imread("back.jpg")
    image_h, image_w, c = back.shape

    centers = {}
    CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 85]]
    CocoPairs = [
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
        (11, 12), (12, 13), (1, 0), (0, 14), (14, 15), (5, 15)
    ]

    for pos in range(0, 16):
        center = (int(test[2 * pos] * (image_w // 2) + 0.5), int(test[2 * pos + 1] * (image_h // 2)))
        centers[pos] = center
        cv2.circle(back, center, 3, CocoColors[pos], thickness=3, lineType=8, shift=0)

    for pair_order, pair in enumerate(CocoPairs):
        cv2.line(back, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return back

def visualize_results(current_state, predicted_state):
    cur = draw(current_state)
    future = draw(predicted_state)
    cv2.imshow("predict future state", future)
    cv2.imshow("input current state", cur)
    return cv2.waitKey(1)