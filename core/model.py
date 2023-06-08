import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

changedict = {5: 1,
              2: 2,
              15: 3,
              9: 4,
              40: 5,
              6: 6,
              3: 7,
              16: 8,
              57: 9,
              20: 10,
              61: 11,
              17: 12,
              18: 13,
              4: 14,
              1: 15,
              59: 16,
              19: 17,
              58: 18,
              7: 19,
              63: 20}


def make_colormap(num=256):
    def bit_get(val, idx):
        return (val >> idx) & 1

    colormap = np.zeros((num, 3), dtype=int)
    ind = np.arange(num, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


idxs = [i for i in changedict]
cmap = make_colormap(256).tolist()
palette = [value for color in cmap for value in color]


class Model:
    def __init__(self, detection, depth, args, colors):
        self.detection = detection(args.od)
        self.depth = depth(args.de)
        self.src = args.source
        self.save = args.save
        self.colors = colors

        self.names = None
        self.original_img = None
        self.depth_image = None
        self.processing_image = None
        self.crop_images = None
        self.ordered_image = None
        self.K_masking_image = None
        self.G_masking_image = None
        self.selected_cluster = None
        self.K_masking_canvas = None
        self.G_masking_canvas = None
        self.masking_canvas = None
        # self.dcrf_image = None
        # self.filtered_dcrf_image = None
        self.K_masking_filtered_crf = None
        self.G_masking_filtered_crf = None
        self.K_masking_crf = None
        self.G_masking_crf = None

        self.filtered_K_masking_image = None
        self.filtered_G_masking_image = None

    def run(self):
        self.inference()
        self.preprocess()
        self.crop()
        self.postprocess()
        self.ordered_paint()
        self.post_crf()
        self.final_filter()

    def inference(self):
        self.detection(self.src, save=self.save)
        self.names = self.detection.result.names
        self.depth(self.src, save=self.save)
        self.original_img = self.detection.result.orig_img

    def preprocess(self):

        depth_img = cv2.cvtColor(self.depth.plot(), cv2.COLOR_BGRA2GRAY)
        self.depth_image = depth_img.copy()
        height, width = depth_img.shape
        kernel = np.zeros((height, width))
        th = int(height * 1)  # <Param 1> Default : 1
        x = np.full((width, th), np.linspace(0, 255, th)).transpose()
        kernel[height - th:, :] = x
        normalized_depth = depth_img - kernel
        normalized_depth = normalized_depth.clip(min=0, max=255)
        self.processing_image = normalized_depth

    def crop(self):
        detections = self.detection.result.boxes.data.detach().cpu().numpy()

        o_depth_img = self.depth_image
        depth_img = self.processing_image
        crop_images = []
        for detection in detections:
            *box, conf, cls = detection
            if (cls + 1) not in changedict:
                continue
            if conf < 0.6:
                continue
            l, t, r, b = map(int, box)
            centroid = ((t + b) // 2, (l + r) // 2)
            crop_images.append(
                dict(cls=int(cls),
                     conf=conf,
                     depth_value=depth_img[centroid],
                     ltrb=(l, t, r, b),
                     centroid=centroid,
                     original_img=self.original_img[t:b, l:r].copy(),
                     gray_img=cv2.cvtColor(self.original_img[t:b, l:r].copy(), cv2.COLOR_BGR2GRAY),
                     norm_depth_img=depth_img[t:b, l:r].astype(np.uint8),
                     depth_img=o_depth_img[t:b, l:r]))

        self.crop_images = sorted(crop_images, key=lambda x: x['depth_value'])

    @staticmethod
    def median(img):
        height, width = img.shape
        out2 = np.zeros((height + 4, width + 4), dtype=float)
        out2[2: 2 + height, 2: 2 + width] = img.copy().astype(float)
        temp2 = out2.copy()

        for i in range(height):
            for j in range(width):
                hybrid_temp1 = np.median((temp2[i, j], temp2[i + 1, j + 1], temp2[i + 2, j + 2],
                                          temp2[i + 3, j + 3], temp2[i + 4, j + 4]))
                hybrid_temp2 = np.median((temp2[i + 4, j], temp2[i + 3, j + 1], temp2[i + 2, j + 2],
                                          temp2[i + 1, j + 3], temp2[i, j + 4]))
                hybrid_temp3 = np.median((temp2[i: i + 5, j:j + 5]))
                out2[2 + i, 2 + j] = np.median((hybrid_temp1, hybrid_temp2, hybrid_temp3))

        out2 = out2[2:2 + height, 2:2 + width].astype(np.uint8)

        return out2

    @staticmethod
    def final_median(img):
        height, width, channel = img.shape

        out2 = np.zeros((height + 4, width + 4, channel), dtype=float)
        out2[2: 2 + height, 2: 2 + width] = img.copy().astype(float)
        temp2 = out2.copy()

        for i in range(height):
            for j in range(width):
                for k in range(channel):
                    hybrid_temp1 = np.median((temp2[i, j, k], temp2[i + 1, j + 1, k], temp2[i + 2, j + 2, k],
                                              temp2[i + 3, j + 3, k], temp2[i + 4, j + 4, k]))
                    hybrid_temp2 = np.median((temp2[i + 4, j, k], temp2[i + 3, j + 1, k], temp2[i + 2, j + 2, k],
                                              temp2[i + 1, j + 3, k], temp2[i, j + 4, k]))
                    hybrid_temp3 = np.median((temp2[i: i + 5, j:j + 5, k]))
                    out2[2 + i, 2 + j, k] = np.median((hybrid_temp1, hybrid_temp2, hybrid_temp3))

        out2 = out2[2:2 + height, 2:2 + width].astype(np.uint8)
        return out2

    def postprocess(self):
        for image in self.crop_images:

            if image['depth_img'].max() < 20:
                continue

            height = image['original_img'].shape[0]
            width = image['original_img'].shape[1]

            # canvas = np.full((height, width, 3), (0, 0, 0), dtype=np.uint8)
            # oi = image['original_img'].copy()
            # de = (image['depth_img'] / 255).reshape(height, width, -1)
            # image['apply_depth'] = (oi * de).astype(np.uint8)
            # canvas[:, :] = image['apply_depth'].copy()

            canvas = np.full((height, width), 0, dtype=np.uint8)
            oi = image['gray_img'].copy()
            de = (image['depth_img'] / 255).reshape(height, width)
            image['apply_depth'] = (oi * de).astype(np.uint8)
            canvas[:, :] = image['apply_depth'].copy()

            kmeans = KMeans(n_clusters=3, n_init='auto')
            gmm = GaussianMixture(n_components=3)

            km_cluster_labels = kmeans.fit_predict(canvas.reshape(-1, 1))
            gmm_cluster_labels = gmm.fit_predict(canvas.reshape(-1, 1))

            km_clustered_image = np.zeros((height, width), dtype=np.uint8)
            gmm_clustered_image = np.zeros((height, width), dtype=np.uint8)

            for i in range(height):
                for j in range(width):
                    km_clustered_image[i, j] = kmeans.cluster_centers_[km_cluster_labels[i * width + j]][0]
                    gmm_clustered_image[i, j] = gmm.means_[gmm_cluster_labels[i * width + j]][0]

            image['K_blur'] = self.median(km_clustered_image)

            image['G_blur'] = self.median(gmm_clustered_image)

            fusion_img = image['K_blur'].astype(int) + image['G_blur'].astype(int)

            image['fusion'] = fusion_img.copy().clip(min=0, max=255).astype(np.uint8)

            #            fusion_img = ((fusion_img - fusion_img.min()) / (fusion_img.max() - fusion_img.min()) * 255).astype(
            #                np.uint8)
            #
            #            image['normalized_fusion'] = fusion_img.copy()
            #
            image['fusion_mean_masking'] = np.where(fusion_img > fusion_img.mean(), fusion_img, 0).astype(np.uint8)
            #
            # image['fmm_depth_fusion'] = (
            #         image['fusion_mean_masking'].copy().astype(int) + image['depth_img'].copy().astype(int)).clip(
            #     min=0, max=255).astype(np.uint8)

            # print('-------------------------------------------------------')
            # print(f"image['fusion_mean_masking'].shape : {type(image['fusion_mean_masking'][0,0])}")
            # print(f"image['norm_depth_img'].shape : {type(image['norm_depth_img'][0,0])}")
            # print('-------------------------------------------------------')
            # exit()

            image['fmm_depth_fusion'] = cv2.addWeighted(image['fusion_mean_masking'].copy(), 0.35,
                                                        image['norm_depth_img'].copy().astype(np.uint8), 1, 0)
            #
            #            cv2.imwrite('fmm.png', image['fmm_depth_fusion'])
            #
            canvas = np.full((height, width), 0, dtype=np.uint8)
            canvas[:, :] = image['fmm_depth_fusion'].copy()

            # canvas = np.full((height, width), 0, dtype=np.uint8)
            # canvas[:, :] = image['fusion'].copy()

            kmeans = KMeans(n_clusters=2, n_init='auto')
            gmm = GaussianMixture(n_components=2)

            km_cluster_labels = kmeans.fit_predict(canvas.reshape(-1, 1))
            gmm_cluster_labels = gmm.fit_predict(canvas.reshape(-1, 1))

            km_clustered_image = np.zeros((height, width), dtype=np.uint8)
            gmm_clustered_image = np.zeros((height, width), dtype=np.uint8)

            for i in range(height):
                for j in range(width):
                    km_clustered_image[i, j] = kmeans.cluster_centers_[km_cluster_labels[i * width + j]]
                    gmm_clustered_image[i, j] = gmm.means_[gmm_cluster_labels[i * width + j]]

            image['F_KMean'] = km_clustered_image.copy()
            image['F_GMM'] = gmm_clustered_image.copy()

            val, cnt = np.unique(image['F_KMean'], return_counts=True)
            vals = {int(k): v for k, v in zip(val, cnt)}
            gv = sorted(vals.keys())
            # print('KMEAN', gv)
            image['F_KMean_mask'] = gv[-1:]

            val, cnt = np.unique(image['F_GMM'], return_counts=True)
            vals = {int(k): v for k, v in zip(val, cnt)}
            gv = sorted(vals.keys())
            # print('GMM', gv)
            image['F_GMM_mask'] = gv[-1:]

    def ordered_paint(self):
        canvas = self.detection.plot(label=False)
        h, w, _ = canvas.shape
        K_masking_canvas = np.zeros((h, w), dtype=np.uint8)
        G_masking_canvas = np.zeros((h, w), dtype=np.uint8)
        for idx, image in enumerate(self.crop_images):
            if image['depth_img'].max() < 20:
                continue
            cls = image['cls']
            color = self.colors(cls, True)
            cv2.putText(canvas, str(idx + 1), image['ltrb'][:2], cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
            l, t, r, b = image['ltrb']
            if image['F_KMean_mask']:
                box = K_masking_canvas[t:b, l:r]
                for g_max in image['F_KMean_mask']:
                    box[image['F_KMean'] == g_max] = changedict[cls + 1]
                K_masking_canvas[t:b, l:r] = box
            if image['F_GMM_mask']:
                box = G_masking_canvas[t:b, l:r]
                for g_max in image['F_GMM_mask']:
                    box[image['F_GMM'] == g_max] = changedict[cls + 1]
                G_masking_canvas[t:b, l:r] = box

        self.K_masking_canvas = K_masking_canvas
        self.G_masking_canvas = G_masking_canvas
        self.masking_canvas = K_masking_canvas

        self.ordered_image = canvas
        self.K_masking_image = Image.fromarray(K_masking_canvas).convert('P')
        self.K_masking_image.putpalette(palette)
        self.G_masking_image = Image.fromarray(G_masking_canvas).convert('P')
        self.G_masking_image.putpalette(palette)

    #        self.K_masking_image = K_masking_canvas
    #        self.G_masking_image = G_masking_canvas

    def _final_filter(self, img):
        # self.filtered_K_masking_image = self.final_median(self.K_masking_image)
        img_png = Image.fromarray(self.median(np.array(img))).convert('P')
        img_png.putpalette(palette)
        return img_png
        # self.filtered_dcrf_image = img_png

    def final_filter(self):
        self.K_masking_filtered_crf = self._final_filter(self.K_masking_crf)
        self.G_masking_filtered_crf = self._final_filter(self.G_masking_crf)

    def post_crf(self):
        self.K_masking_crf = self._post_crf(self.K_masking_canvas)
        self.G_masking_crf = self._post_crf(self.G_masking_canvas)

    def _post_crf(self, img):
        # img = self.masking_canvas
        h, w = img.shape
        dcrf_model = dcrf.DenseCRF2D(w, h, 21)

        unary = unary_from_labels(img, 21, gt_prob=0.65, zero_unsure=False)

        dcrf_model.setUnaryEnergy(unary)
        dcrf_model.addPairwiseGaussian(sxy=(3, 3), compat=3)
        dcrf_model.addPairwiseBilateral(sxy=80, srgb=13, rgbim=self.original_img, compat=10)

        Q = dcrf_model.inference(5)
        _map = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8)
        img_png = Image.fromarray(_map).convert('P')
        img_png.putpalette(palette)
        # self.dcrf_image = img_png
        return img_png