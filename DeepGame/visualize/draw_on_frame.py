from __future__ import division
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils

[W,H] = utils.get_img_dim()
num_tiles = utils.get_num_tiles()

def pred_to_box(arr):
    rects =[]
    for i in range(len(arr)):
        if arr[i]:
            tile_w = W/num_tiles
            tile_h = H/num_tiles
            x = i*tile_w
            y = i*tile_h
            y = 3*tile_h
            new_rectangle = [x,y,tile_w,tile_h]
            rects.append(new_rectangle)
    return rects

if __name__ == "__main__":
    #gt_circle = [0.58645,0.40389]
    gt_circle = [0.5761,0.43268]
    gt_rect1 = [0,0,0,1,1,0,0]
    gt_rect2 = [0,0,0,0,1,0,0]
    os.makedirs("output", exist_ok=True)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    img_path = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\selected_frames\\ha_0\\frame_00700.png'
    print("Image: '%s'" % (img_path))
    img = np.array(Image.open(img_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    rectangles = pred_to_box(gt_rect1)
    for i in range(len(rectangles)):
        color = colors[0]
        x1 = rectangles[i][0]
        y1 = rectangles[i][1]
        box_w = rectangles[i][2]
        box_h = rectangles[i][3]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(bbox)
    rectangles_gt = pred_to_box(gt_rect2)
    for i in range(len(rectangles_gt)):
        color = colors[16]
        x1 = rectangles_gt[i][0]
        y1 = rectangles_gt[i][1]
        box_w = rectangles_gt[i][2]
        box_h = rectangles_gt[i][3]
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(bbox)
    # Create a Rectangle patch
    color = colors[-1]
    roi_circle  = patches.Circle((gt_circle[0]*W, gt_circle[1]*H), radius=75, edgecolor=color, facecolor="none")
    # Add the bbox to the plot
    ax.add_patch(roi_circle)
    # Add label


    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    print(img_path)
    print(img_path.split("/")[-1].split(".")[0])
    filename = img_path.split("\\")[-1].split(".")[0]
    plt.savefig(f"output\\{filename}.png", bbox_inches="tight", pad_inches=0.0)
    plt.close()
