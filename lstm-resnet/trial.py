import glob
import numpy as np
ha_0_images = sorted(glob.glob("/home/omossad/scratch/temp/roi/images/ha_0_*"))
ha_0_labels = sorted(glob.glob("/home/omossad/scratch/temp/roi/labels/ha_0_*"))
print(len(ha_0_images))
print(len(ha_0_labels))


time_steps = 4

def closestNumber(n, m) :
    q = int(n / m)
    return int(q*m)




def process_data(images):
    num_images = len(images)
    image_indices = np.arange(0,num_images)
    indices = np.array([ image_indices[i:i+time_steps] for i in range(num_images-time_steps) ])
    images=np.asarray(images)
    print(images[indices])

process_data(ha_0_images)
