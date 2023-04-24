class Model:
    def __init__(self, detection, depth, args):
        self.detection = detection(args.od)
        self.depth = depth(args.de)
        self.src = args.source
        self.save = args.save

        self.names = None
        self.crop_images = None

    def inference(self):
        self.detection(self.src, save=self.save)
        self.names = self.detection.result.names
        self.depth(self.src, save=self.save)

    def crop(self):
        detections = self.detection.result.boxes.data.detach().cpu().numpy()
        original_img = self.detection.result.orig_img
        depth_img = self.depth.plot()
        crop_images = []
        for detection in detections:
            *box, conf, cls = detection
            l, t, r, b = map(int, box)
            crop_images.append(
                dict(cls=int(cls), conf=conf, original_img=original_img[t:b, l:r], depth_img=depth_img[t:b, l:r]))
        self.crop_images = crop_images


