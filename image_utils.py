import imageio
import numpy as np
import os
import struct


def write_image_imageio(img_file, img, quality):
    img = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    kwargs = {}
    if os.path.splitext(img_file)[1].lower() in [".jpg", ".jpeg"]:
        if img.ndim >= 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        kwargs["quality"] = quality
        kwargs["subsampling"] = 0
    imageio.imwrite(img_file, img, **kwargs)


def read_image_imageio(img_file):
    img = imageio.imread(img_file, pilmode="RGB")
    img = np.asarray(img).astype(np.float32)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img / 255.0


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (np.sign(img) * (np.abs(img)) ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def read_image(file):
    if os.path.splitext(file)[1] == ".bin":
        with open(file, "rb") as f:
            bytes = f.read()
            h, w = struct.unpack("ii", bytes[:8])
            img = np.frombuffer(bytes, dtype=np.float16, count=h * w * 4, offset=8).astype(np.float32).reshape(
                [h, w, 4])
    else:
        img = read_image_imageio(file)
        if img.shape[2] == 4:
            img[..., 0:3] = srgb_to_linear(img[..., 0:3])
            # Premultiply alpha
            img[..., 0:3] *= img[..., 3:4]
        else:
            img = srgb_to_linear(img)
    return img


def write_image(file, img, quality=95):
    if os.path.splitext(file)[1] == ".bin":
        if img.shape[2] < 4:
            img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
        with open(file, "wb") as f:
            f.write(struct.pack("ii", img.shape[0], img.shape[1]))
            f.write(img.astype(np.float16).tobytes())
    else:
        if img.shape[2] == 4:
            img = np.copy(img)
            # Unmultiply alpha
            img[..., 0:3] = np.divide(img[..., 0:3], img[..., 3:4], out=np.zeros_like(img[..., 0:3]),
                                      where=img[..., 3:4] != 0)
            img[..., 0:3] = linear_to_srgb(img[..., 0:3])
        else:
            img = linear_to_srgb(img)
        write_image_imageio(file, img, quality)


