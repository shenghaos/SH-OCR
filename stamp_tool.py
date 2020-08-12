import cv2 as cv
import numpy as np


def delete_stamp(img, thr=100, point=(-1, -1)):
    if isinstance(img, str):
        img = cv.imread(img)
    assert img is not None
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    h, s, v = cv.split(hsv)
    v[v > thr] = 255
    v[v <= thr] = 0

    if point != (-1, -1):
        v = 255 - v

        h, w, c = img.shape
        point = list(point)

        assert point[0] < w and point[1] < h

        b, g, r = cv.split(img)

        b_p = b[point[0]][point[1]]
        g_p = g[point[0]][point[1]]
        r_p = r[point[0]][point[1]]
        b_c = np.zeros((h, w), dtype=np.uint8)
        g_c = np.zeros((h, w), dtype=np.uint8)
        r_c = np.zeros((h, w), dtype=np.uint8)

        b_p = b_c + b_p
        g_p = g_c + g_p
        r_p = r_c + r_p

        new_b = cv.subtract(b_p, v)
        new_g = cv.subtract(g_p, v)
        new_r = cv.subtract(r_p, v)

        dst = cv.merge([new_b, new_g, new_r])
        return dst
    dst = cv.merge([v, v, v])
    return dst


if __name__ == "__main__":
    dst = delete_stamp(r'D:\OCR\Logo\timg2.jpg', point=(0, 0))
    cv.imshow('r', dst)
    cv.waitKey(0)
