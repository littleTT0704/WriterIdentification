import cv2
from lime import lime_image
import numpy as np
from skimage import color
from skimage.segmentation import mark_boundaries


def interpret(m):
    def pred_fn(images):
        pred = np.array([255 - color.rgb2gray(img) for img in images])
        return m.predict(pred)[0]

    explainer = lime_image.LimeImageExplainer()

    with open("./log/box.txt", "w") as f:
        s = eval(f.readline())
        n = len(s)

    for i in range(n):
        img = cv2.imread("./log/%d.png" % i)
        preds = pred_fn(np.array([img]))
        x = preds.argsort()[0][-1]
        explanation = explainer.explain_instance(
            img, pred_fn, top_labels=5, hide_color=255, num_samples=1000
        )
        temp, mask = explanation.get_image_and_mask(
            x, positive_only=False, num_features=5, hide_rest=False
        )
        boundaries = mark_boundaries(temp / 2 + 0.5, mask, color=(0, 255, 255))
        cv2.imwrite("./log/%d_exp.png" % i, boundaries)

        img = cv2.imread("./data/img/%d/%s.png" % (i, s[i]))
        cv2.imwrite("./log/%d_sim.png", img)
