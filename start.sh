mkdir storage
mkdir ../../datasets/
rm -r ../sample_data

cp -r /content/drive/MyDrive/Colab/ReID\ works/CVPR\ fintuning/net_149.pth ./storage
cp -r /content/drive/MyDrive/科研数据集/prid2011.zip ../../datasets/
unzip ../../datasets/prid2011.zip -d ../../datasets