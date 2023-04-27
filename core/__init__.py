import cv2
import numpy as np


class Model:
    def __init__(self, detection, depth, args, colors):
        self.detection = detection(args.od)
        self.depth = depth(args.de)
        self.src = args.source
        self.save = args.save
        self.colors = colors

        self.names = None
        self.original_img = None
        self.gray_image = None
        self.blur_image = None
        self.thresh_image = None
        self.edge = None
        self.closed_edge_image = None
        self.contours_image = None
        self.processing_image = None
        self.crop_images = None
        self.ordered_image = None
        self.masking_image = None

    def inference(self):
        self.detection(self.src, save=self.save)
        self.names = self.detection.result.names
        self.depth(self.src, save=self.save)

    def crop(self):
        detections = self.detection.result.boxes.data.detach().cpu().numpy()

        depth_img = self.processing_image
        crop_images = []
        for detection in detections:
            *box, conf, cls = detection
            l, t, r, b = map(int, box)
            centroid = ((t + b) // 2, (l + r) // 2)
            crop_images.append(
                dict(cls=int(cls),
                     conf=conf,
                     depth_value=depth_img[centroid],
                     ltrb=(l, t, r, b),
                     original_img=self.original_img[t:b, l:r],
                     depth_img=depth_img[t:b, l:r]))

        self.crop_images = sorted(crop_images, key=lambda x: x['depth_value'])

    def preprocess(self):
        self.original_img = self.detection.result.orig_img
        # self.gray_image = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)
        # self.blur_image = cv2.GaussianBlur(self.gray_image, ksize=(5, 5), sigmaX=0)
        # _, self.thresh_image = cv2.threshold(self.blur_image, 127, 255, cv2.THRESH_BINARY)
        # self.edge = cv2.Canny(self.blur_image, 10, 250)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        # self.closed_edge_image = cv2.morphologyEx(self.edge, cv2.MORPH_CLOSE, kernel)
        # contours, _ = cv2.findContours(self.closed_edge_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # self.contours_image = cv2.drawContours(self.original_img, contours, -1, (0, 255, 0), 3)

        processing_image = cv2.cvtColor(self.depth.plot(), cv2.COLOR_BGRA2GRAY)
        height, width = processing_image.shape
        kernel = np.zeros((height, width))
        th = int(height * 1)  # <Param 1> Default : 1
        x = np.full((width, th), np.linspace(0, 255, th)).transpose()
        kernel[height - th:, :] = x
        normalized_depth = processing_image - kernel
        normalized_depth = normalized_depth.clip(min=0, max=255)
        self.processing_image = normalized_depth

    def postprocess(self):
        for image in self.crop_images:
            cls = image['cls']
            color = self.colors(cls, True)

            threshold = int(image['depth_value'] * 0.8)  # <Param 2> Default : 0.5
            ret, output = cv2.threshold(image['depth_img'], threshold, 255, cv2.THRESH_BINARY)
            image['mask'] = output

            gray = cv2.cvtColor(image['original_img'], cv2.COLOR_BGR2GRAY)
            image['gray'] = gray
            blur = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
            image['blur'] = blur
            _, thresh_image = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
            image['thresh_image'] = thresh_image
            edge = cv2.Canny(blur, 10, 250)
            image['edge'] = edge
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed_edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
            image['closed_edge'] = closed_edge
            contours, _ = cv2.findContours(closed_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_image = cv2.drawContours(image['original_img'], contours, -1, color, 3)
            image['contours_image'] = contours_image

        # cv2.imwrite('canny.png', cv2.Canny(cv2.imread('cat_0001_depth.png'), 30, 30))
        # cv2.imwrite('canny.png', cv2.Canny(cv2.imread('normalized_depth_output.png'), 30, 30))
        # cv2.imwrite('canny_origin.png', cv2.Canny(cv2.imread('cat_0001_origin.png'), 150, 150))
        # cv2.imwrite('add.png', cv2.addWeighted(cv2.imread('canny_whole.png'), 1, cv2.imread('masking_image.png'), 1, 0))

    def ordered_paint(self):
        canvas = self.detection.plot(label=False)
        masking_canvas = np.zeros(canvas.shape, dtype=np.uint8)
        for idx, image in enumerate(self.crop_images):
            cls = image['cls']
            color = self.colors(cls, True)
            cv2.putText(canvas, str(idx + 1), image['ltrb'][:2], cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            l, t, r, b = image['ltrb']
            box = masking_canvas[t:b, l:r]
            box[image['mask'] > 0] = color
            masking_canvas[t:b, l:r] = box
        self.ordered_image = canvas
        self.masking_image = masking_canvas
