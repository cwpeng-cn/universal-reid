import csv
from glob import glob
import os.path as osp

dataset_path = "../../datasets/prid2011/prid_2011/multi_shot/"

txt_name = "pseudo_label_prid.txt"
csv_name = 'pseudo_label_prid.csv'
out = open(csv_name, 'a', newline='')

# 设定写入模式
csv_write = csv.writer(out, dialect='excel')
with open(txt_name, 'r') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.split("\n")[0]
        cama_person, camb_person = line.split("\t")
        cama_person_images = sorted(glob(osp.join(dataset_path + cama_person, '*.png')))
        camb_person_images = sorted(glob(osp.join(dataset_path + camb_person, '*.png')))
        content = cama_person_images + camb_person_images
        csv_write.writerow(content)
print("write over")
