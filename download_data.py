import os
import gdown
import zipfile
import numpy as np
import skimage.io
import shutil

# video data from GoPro540p
data_url = "https://drive.google.com/uc?id=1tBi08YWAtOxjAmMkmymb0U9IRaXcZ3E1"
save_dir = "./data/clean"

# 1. Download the zip file
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
zip_path = os.path.join(save_dir, "gopro_540p.zip")
gdown.download(data_url, output=zip_path, quiet=False)

# 2. Extract the zip file
print("Extracting ...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(save_dir)
print("Extraction complete.")

# 3. Convert to TIF format
video_dir = os.path.join(save_dir, "gopro_540p")
for video_name in os.listdir(video_dir):
    print(f"Converting {video_name} to TIF format ...")
    video_frame_dir = os.path.join(video_dir, video_name)
    video_frame_list = list(os.listdir(video_frame_dir))
    video_frame_list = [_ for _ in video_frame_list if _.endswith('.png')]
    video_frame_list.sort(key=lambda x: int(x.split('.')[0]))
    # print(f'video_frame_list: {video_frame_list}')
    print(f'num frames: {len(video_frame_list)}')
    data = []
    for i, frame_name in enumerate(video_frame_list):
        frame_path = os.path.join(video_frame_dir, frame_name)
        frame = skimage.io.imread(frame_path)
        data.append(frame)
    data = np.stack(data, axis=0)  # [N, H, W, C]
    print(f'data.shape: {data.shape}, data.dtype: {data.dtype}')

    # save
    tif_path = os.path.join(save_dir, video_name + '.tif')
    skimage.io.imsave(tif_path, data)

    print(f"Saved {video_name} as TIF.")

# 4. Remove the original video directory
shutil.rmtree(video_dir)
    