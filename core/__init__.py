import cv2
import numpy as np


class Model:
    def __init__(self, detection, depth, args, colors):
        self.detection = detection(args.od)
        self.depth = depth(args.de)
        self.src = args.source
        self.save = args.save
        self.colors = colors

        self.point_interval = 20

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
        processing_image = cv2.cvtColor(self.depth.plot(), cv2.COLOR_BGRA2GRAY)
        height, width = processing_image.shape
        kernel = np.zeros((height, width))
        th = int(height * 1)  # <Param 1> Default : 1
        x = np.full((width, th), np.linspace(0, 255, th)).transpose()
        kernel[height - th:, :] = x
        normalized_depth = processing_image - kernel
        normalized_depth = normalized_depth.clip(min=0, max=255)
        self.processing_image = normalized_depth
        # for p in normalized_depth.astype(np.uint8):
        #     for q in p:
        #         if q > 99:
        #             print('---',end=' ')
        #         else:
        #             z = str(q).zfill(3)
        #
        #             # print(str(q).rjust(3, ' '), end='')
        #
        #             print('   ' if z == '000' else z, end=' ')
        #     print()

    def postprocess(self):
        for image in self.crop_images:
            cls = image['cls']
            color = self.colors(cls, True)
            depth_img = image['depth_img'].copy().astype(np.uint8)
            h, w = depth_img.shape
            indices = np.array(
                [[i, j] for i in range(self.point_interval, h - self.point_interval, self.point_interval) for j in
                 range(self.point_interval, w - self.point_interval, self.point_interval)])
            bgd_model = np.zeros((1, 65), np.float64)  # 배경 모델 무조건 1행 65열, float64
            fgd_model = np.zeros((1, 65), np.float64)  # 전경 모델 무조건 1행 65열,

            # _, mask = cv2.threshold(depth_img, 1, 255, cv2.THRESH_BINARY)
            # mask[mask == 255] = 1

            mask = np.zeros_like(depth_img, np.uint8)
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
            rc = (1, 1, w - 1, h - 1)
            cv2.grabCut(depth_img, mask, rc, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
            depth_img = depth_img * mask2[:, :, np.newaxis]
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGRA2GRAY)

            # proc_coords = []
            for i in range(1, 20):
                pointed_depth_img = depth_img[indices[:, 0], indices[:, 1]]

                p_max = indices[
                    np.argwhere(pointed_depth_img == np.max(pointed_depth_img[pointed_depth_img != 255], axis=0))]
                p_min = indices[
                    np.argwhere(pointed_depth_img == np.min(pointed_depth_img[pointed_depth_img != 0], axis=0))]

                for y, x in p_max[0]:
                    # if [y,x] not in proc_coords:
                    cv2.circle(mask, (x, y), 3, cv2.GC_FGD, -1)
                    # proc_coords.append([y,x])
                    cv2.circle(depth_img, (x, y), 3, 255, -1)
                for y, x in p_min[0]:
                    # if [y, x] not in proc_coords:
                    cv2.circle(mask, (x, y), 3, cv2.GC_BGD, -1)
                    # proc_coords.append([y, x])
                    cv2.circle(depth_img, (x, y), 3, 0, -1)

                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2BGR)
                cv2.grabCut(depth_img, mask, rc, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
                # mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype(np.uint8)
                mask2 = np.where((mask == 0) + (mask == 2), 0, 1).astype(np.uint8)
                depth_img = depth_img * mask2[:, :, np.newaxis]
                depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGRA2GRAY)
                # depth_img = cv2.bitwise_and(depth_img, mask2)

                cv2.imwrite(f'pointed_depth_img-{i}.png', depth_img)

                _, mask3 = cv2.threshold(depth_img, 1, 255, cv2.THRESH_BINARY)
                cv2.imwrite(f'mask-{i}.png', mask3)
                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                # mask[mask == 255] = 1

            mask[mask == 1] = 255
            image['mask'] = mask

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
