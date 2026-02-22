from __future__ import division
import numpy as np
import os
import sys
import random
from matplotlib.pyplot import imsave
import matplotlib
matplotlib.use("Agg")
from PIL import Image

# --- helpers 16-bit ---

def ensure_uint16(arr: np.ndarray) -> np.ndarray:
    """Force uint16 without copying if already ok."""
    if arr.dtype == np.uint16:
        return arr
    # if old uint8 data appears, you may want to upscale; but here we keep logic simple:
    return arr.astype(np.uint16, copy=False)

def to_pil_16(arr16: np.ndarray) -> Image.Image:
    """Convert uint16 numpy array (H,W) -> PIL Image mode I;16."""
    arr16 = ensure_uint16(arr16)
    return Image.fromarray(arr16, mode="I;16")

def from_pil_16(img: Image.Image) -> np.ndarray:
    """Convert PIL Image back to uint16 numpy array."""
    arr = np.asarray(img)
    # Some PIL ops may yield int32; clamp back to uint16 range
    if arr.dtype != np.uint16:
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    return arr

def vis_uint16_to_uint8(arr16: np.ndarray) -> np.ndarray:
    """For visualization only: map uint16 [0..65535] -> uint8 [0..255]."""
    arr16 = ensure_uint16(arr16)
    return (arr16.astype(np.float32) / 65535.0 * 255.0).round().astype(np.uint8)

# --- original logic ---

def random_flip(input, axis, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            axis += 1
        return np.flip(input, axis=axis)
    else:
        return input

def random_crop(input, with_fa=False):
    ran = random.random()
    if ran > 0.2:
        if with_fa:
            rx = int(random.random() * input.shape[1] // 10)
            ry = int(random.random() * input.shape[2] // 10)
            return input[:, rx: rx + int(input.shape[1] * 9 // 10), ry: ry + int(input.shape[2] * 9 // 10)]
        else:
            rx = int(random.random() * input.shape[0] // 10)
            ry = int(random.random() * input.shape[1] // 10)
            return input[rx: rx + int(input.shape[0] * 9 // 10), ry: ry + int(input.shape[1] * 9 // 10)]
    else:
        return input

def random_rotate_90(input, with_fa=False):
    ran = random.random()
    if ran > 0.5:
        if with_fa:
            return np.rot90(input, axes=(1, 2))
        return np.rot90(input)
    else:
        return input

def random_rotation(x, chance, with_fa=False):
    ran = random.random()

    if with_fa:
        # x[0] is image, x[1] is mask
        img = to_pil_16(x[0])
        mask = Image.fromarray(x[1])  # mask logic unchanged; assume mask is not 16-bit intensity
        if ran > 1 - chance:
            angle = np.random.randint(0, 90)
            img = img.rotate(angle=angle, expand=1)
            mask = mask.rotate(angle=angle, expand=1, fillcolor=1)
        return np.stack([from_pil_16(img), np.asarray(mask)])

    # normal image (H,W)
    img = to_pil_16(x)
    if ran > 1 - chance:
        angle = np.random.randint(0, 90)
        img = img.rotate(angle=angle, expand=1)
    return from_pil_16(img)

def augment_numpy_images(path, targetNumber, targetDir, skip=None, rot=True, with_fa=False):
    classes = os.listdir(path)
    if not os.path.exists(targetDir):
        os.mkdir(targetDir)
    for class_ in classes:
        if not os.path.exists(targetDir + class_):
            os.makedirs(targetDir + class_)

    for class_ in classes:
        count, round_ = 0, 0
        while count < targetNumber:
            round_ += 1
            for root, dir, files in os.walk(os.path.join(path, class_)):
                for file in files:
                    if skip and skip in file:
                        continue

                    filepath = os.path.join(root, file)
                    arr = np.load(filepath)
                    arr = ensure_uint16(arr)  # <-- keep everything in uint16

                    print("loaded ", file)
                    print(arr.shape, arr.dtype)

                    try:
                        arr = random_crop(arr, with_fa)

                        if rot:
                            arr = random_rotation(arr, 0.9, with_fa)

                        arr = random_flip(arr, 0, with_fa)
                        arr = random_flip(arr, 1, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        arr = random_rotate_90(arr, with_fa)
                        arr = random_rotate_90(arr, with_fa)

                        # --- same illegal-content logic, unchanged ---
                        if with_fa:
                            whites = arr.shape[2] * arr.shape[1] - np.count_nonzero(np.round(arr[0] - np.amax(arr[0]), 2))
                            black = arr.shape[2] * arr.shape[1] - np.count_nonzero(np.round(arr[0], 2))
                            if arr.shape[2] < 10 or arr.shape[1] < 10 or black >= arr.shape[2] * arr.shape[1] * 0.8 or \
                                whites >= arr.shape[2] * arr.shape[1] * 0.8:
                                print("illegal content")
                                continue
                        else:
                            whites = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr - np.amax(arr), 2))
                            black = arr.shape[0] * arr.shape[1] - np.count_nonzero(np.round(arr, 2))
                            if arr.shape[0] < 10 or arr.shape[1] < 10 or black >= arr.shape[0] * arr.shape[1] * 0.8 or \
                                whites >= arr.shape[0] * arr.shape[1] * 0.8:
                                print("illegal content")
                                continue

                        # visualization every 10
                        if count % 10 == 0:
                            vis_dir = "./visualizations_of_augmentation/" + class_ + "/"
                            if not os.path.exists(vis_dir):
                                os.makedirs(vis_dir)

                            if with_fa:
                                # image arr[0] is uint16 -> convert to uint8 only for saving PNG
                                im0 = vis_uint16_to_uint8(arr[0])
                                # mask left as-is; if it’s not uint8 you may want to cast for imsave
                                mk = arr[1]
                                if mk.dtype != np.uint8:
                                    mk = mk.astype(np.uint8, copy=False)
                                rgb = np.transpose(np.stack([im0, im0, mk]), (1, 2, 0))
                                imsave(vis_dir + str(count) + ".png", rgb)
                            else:
                                im = vis_uint16_to_uint8(arr)
                                rgb = np.transpose(np.stack([im, im, im]), (1, 2, 0))
                                imsave(vis_dir + str(count) + ".png", rgb)

                        # save augmented npy in uint16
                        out_path = targetDir + class_ + "/" + file[:-4] + "aug" + str(round_)
                        np.save(out_path, ensure_uint16(arr))

                        count += 1
                        print(count)

                    except Exception:
                        print("something is wrong in try, details:", sys.exc_info()[2])
                        err_dir = "./error_of_augmentation/" + class_ + "/"
                        if not os.path.exists(err_dir):
                            os.makedirs(err_dir)
                        np.save(err_dir + str(count), ensure_uint16(arr))

                    if count > targetNumber:
                        break
    print(count)

if __name__ == "__main__":
    print("Data augmentation (uint16)")
    for pos in ["Spiculated", "Circumscribed", "Indistinct"]:
        augment_numpy_images(
            path="/media/grains6lab2/d/iaiabl_my/CBIS_IABL/train",
            targetNumber=250,
            targetDir="/media/grains6lab2/d/iaiabl_my/CBIS_IABL/train_augmented_250/",
            rot=True,
            with_fa=False
        )