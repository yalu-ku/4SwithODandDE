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
        original_img = self.detection.result.orig_img
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
                     original_img=original_img[t:b, l:r],
                     depth_img=depth_img[t:b, l:r]))

        self.crop_images = sorted(crop_images, key=lambda x: x['depth_value'])

    def preprocess(self):
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
            threshold = int(image['depth_value'] * 0.5)  # <Param 2> Default : 0.5
            ret, output = cv2.threshold(image['depth_img'], threshold, 255, cv2.THRESH_BINARY)
            image['mask'] = output

    def ordered_paint(self):
        canvas = self.detection.plot(label=False)
        masking_canvas = np.zeros(canvas.shape, dtype=np.uint8)
        for idx, image in enumerate(self.crop_images):
            cls = image['cls']
            color = self.colors(cls, True)
            cv2.putText(canvas, str(idx + 1), image['ltrb'][:2], cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            l,t,r,b = image['ltrb']
            box = masking_canvas[t:b,l:r]
            box[image['mask'] > 0] = color
            masking_canvas[t:b, l:r] = box
        self.ordered_image = canvas
        self.masking_image = masking_canvas
