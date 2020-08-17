import cv2 as cv
import numpy as np
import copy


def adaptive_delete_stamp(img, point=(), k=1.2):
    assert k >= 1, 'the value of k is wrong!'

    if isinstance(img, str):
        img = cv.imread(img)
    assert img is not None, 'load img failed !'
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    b, g, r = cv.split(img)

    _h, s, v = cv.split(hsv)
    h, w, c = img.shape

    gr = v[1][1]

    sub = cv.subtract(r, b)

    sub = cv.blur(sub, (3, 3))

    thr, s_sub = cv.threshold(sub, 0, 255, cv.THRESH_OTSU)

    V = copy.deepcopy(v)

    v[s_sub > 0] = gr

    thr, v_sub = cv.threshold(v, 0, 255, cv.THRESH_OTSU)

    v_sub = 255 - v_sub

    v_charactor = cv.bitwise_and(v, v, mask=v_sub)

    num = np.sum(v_sub) / 255

    ss = int(np.sum(v_charactor) / num * k)

    end = cv.threshold(V, ss, 255, cv.THRESH_BINARY)[1]

    # cv.imshow('v', end)
    # cv.waitKey(0)

    if point != ():
        end = 255 - end

        point = list(point)

        assert point[0] < h and point[0] > 0 and point[1] < h and point[1] > 0, 'point is wrong!'

        b_p = b[point[0]][point[1]]
        g_p = g[point[0]][point[1]]
        r_p = r[point[0]][point[1]]

        b_c = np.zeros((h, w), dtype=np.uint8)
        g_c = np.zeros((h, w), dtype=np.uint8)
        r_c = np.zeros((h, w), dtype=np.uint8)

        b_p = b_c + b_p
        g_p = g_c + g_p
        r_p = r_c + r_p

        new_b = cv.subtract(b_p, end)
        new_g = cv.subtract(g_p, end)
        new_r = cv.subtract(r_p, end)

        dst = cv.merge([new_b, new_g, new_r])
        return dst
    dst = cv.merge([end, end, end])
    return dst


def detect_and_delete_stamp(img, stamp_d=80, k=1):
    if isinstance(img, str):
        img = cv.imread(img)
    assert img is not None, 'load image failed!'

    b, g, r = cv.split(img)

    __h, __s, v = cv.split(cv.cvtColor(img, cv.COLOR_BGR2HSV))

    V = copy.deepcopy(v)

    gr = v[1][1]

    sub = cv.subtract(r, b)

    sub = cv.blur(sub, (3, 3))

    thr, s_sub = cv.threshold(sub, 20, 255, cv.THRESH_BINARY)

    v[s_sub > 0] = gr

    thr, v_sub = cv.threshold(v, 0, 255, cv.THRESH_OTSU)

    v_sub = 255 - v_sub

    v_charactor = cv.bitwise_and(v, v, mask=v_sub)

    num = np.sum(v_sub) / 255

    ss = int(np.sum(v_charactor) / num * k)

    # cv.imshow('g', s_sub)
    # cv.waitKey(0)

    __h, cts, _ = cv.findContours(s_sub, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    boundary = []

    for ct in cts:
        if len(ct) > 100:
            center, length, angle = cv.fitEllipse(ct)
            a, b = length
            if min(a, b) > 20 and max(a, b) < min(a, b) * 1.2 and min(a, b) > stamp_d:
                boundary.append(cv.boundingRect(ct))
        else:
            continue
    assert boundary != [], 'detect stamp failed! you may adapt the value of stamp_p!'

    for b in boundary:
        x, y, w, h = b

        y = y if y >= 2 else 2
        x = x if x >= 2 else 2

        new_area = V[y - 2:y + h + 2, x - 2:x + w + 2]

        new_area = cv.threshold(new_area, ss, 255, cv.THRESH_BINARY)[1]

        new = cv.merge([new_area, new_area, new_area])
        img[y - 2:y + h + 2, x - 2:x + w + 2] = new

    return img


def delete_stamp(img, thr=120, point=(-1, -1)):
    if isinstance(img, str):
        img = cv.imread(img)
    assert img is not None
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    b, g, r = cv.split(img)

    h, s, v = cv.split(hsv)

    v[v > thr] = 255
    v[v <= thr] = 0

    if point != (-1, -1):
        v = 255 - v

        h, w, c = img.shape
        point = list(point)

        assert point[0] < w and point[1] < h

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
    dst = detect_and_delete_stamp(r'D:\OCR\Logo\u0.jpg',  k=1.3)
    cv.imwrite(r'dst44.jpg', dst)
    cv.imshow('r', dst)
    cv.waitKey(0)
