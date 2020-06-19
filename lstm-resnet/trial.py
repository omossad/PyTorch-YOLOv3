import glob
ha_0_images = sorted(glob.glob("/home/omossad/scratch/temp/roi/images/ha_0_*"))
ha_0_labels = sorted(glob.glob("/home/omossad/scratch/temp/roi/labels/ha_0_*"))
print(len(ha_0_images))
print(len(ha_0_labels))
