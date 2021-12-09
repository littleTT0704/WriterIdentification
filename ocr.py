import cv2
import pytesseract
import numpy as np
from skimage import color
from skimage.filters import threshold_yen
from skimage.morphology import opening
from skimage.measure import label


def drawBox(img, b, n=2):
    area = img[b[1] : b[3], b[0] : b[2]]
    rect = np.zeros(area.shape, dtype=np.uint8)
    rect[:, :, n] = 255
    rect = cv2.addWeighted(area, 0.8, rect, 0.2, 1.0)
    img[b[1] : b[3], b[0] : b[2]] = rect
    color = [0, 0, 0]
    color[n] = 255
    img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color)
    return img


def boxes(path):
    img = cv2.imread(path)
    h, w, c = img.shape
    S = 48

    new = 255 - color.rgb2gray(img)
    threshold = threshold_yen(new)
    new = (new > threshold) * 255
    new = opening(new)
    labeled, n = label(new, return_num=True)

    centers = []
    for i in range(1, n):
        y, x = np.where(labeled == i)
        if np.max(y) - np.min(y) > S or np.max(x) - np.min(x) > S:
            cx = (np.max(x) + np.min(x)) // 2
            cy = (np.max(y) + np.min(y)) // 2
            centers.append((cx, cy))

    centers.sort()
    merged = [centers[0]]
    for i in range(1, len(centers)):
        if centers[i][0] - merged[-1][0] < S:
            merged = merged[:-1] + [
                (
                    (centers[i][0] + merged[-1][0]) // 2,
                    (centers[i][1] + merged[-1][1]) // 2,
                )
            ]
        else:
            merged.append(centers[i])
    centers = [(max(min(cx, w - S), S), max(min(cy, h - S), S)) for cx, cy in merged]

    for i, (cx, cy) in enumerate(centers):
        cv2.imwrite("./log/%d.png" % i, img[cy - S : cy + S, cx - S : cx + S])
    for i, (cx, cy) in enumerate(centers):
        n = 2 if i == 6 else 1
        img = drawBox(img, (cx - S, cy - S, cx + S, cy + S), n)
        img = cv2.putText(
            img, str(i), (cx, cy + S), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0)
        )

    cv2.imwrite("./log/ocr.png", img)
    with open("./log/box.txt", "w") as f:
        f.write(str(centers))


if __name__ == "__main__":
    img = cv2.imread("./data/s/0_alt.png")
    boxes = pytesseract.image_to_boxes(img, config=r"-l chi_sim --psm 6")
    n = 0
    for b in boxes.splitlines():
        b = list(map(int, b.split()[1:]))
        img = drawBox(img, b, n % 3)
        n += 1
    cv2.imwrite("./tmp.png", img)
    # img = cv2.rectangle(img, (b[0], h - b[1]), (b[2], h - b[3]), color, 2)
