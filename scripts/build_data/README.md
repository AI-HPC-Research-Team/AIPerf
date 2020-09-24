#### 数据集制作

**数据集下载**

 *Imagenet官方地址：http://www.image-net.org/index* 

官方提供四种数据集：  Flowers、CIFAR-10、MNIST、ImageNet-2012  前三个数据集数据量小，直接调用相关脚本自动会完成下载、转换（TFRecord格式）的过程，在 /userhome/AIPerf/scripts/build_data目录下执行以下脚本：

```javascript
cd  /userhome/AIPerf/scripts/build_data
./download_imagenet.sh
```

原始的ImageNet-2012下载到当前的imagenet目录并包含以下两个文件:

- ILSVRC2012_img_val.tar
- ILSVRC2012_img_train.tar

**TFReord制作**

训练集和验证集需要按照1000个子目录下包含图片的格式，处理步骤：

1. 将train 和 val 的数据按照文件夹分类
2. 指定参数运行build_imagenet_data.py

**可以按照以下步骤执行**:  假设数据存放在/userhome/AIPerf/scripts/build_data/imagenet目录下，TFRecord文件的输出目录是/userhome/AIPerf/scripts/build_data/ILSVRC2012/output

```shell
# 做验证集
cd  /userhome/AIPerf/scripts/build_data
mkdir -p ILSVRC2012/raw-data/imagenet-data/validation/  
tar -xvf imagenet/ILSVRC2012_img_val.tar -C ILSVRC2012/raw-data/imagenet-data/validation/
python preprocess_imagenet_validation_data.py ILSVRC2012/raw-data/imagenet-data/validation/ imagenet_2012_validation_synset_labels.txt

# 做训练集
mkdir -p ILSVRC2012/raw-data/imagenet-data/train/
tar -xvf imagenet/ILSVRC2012_img_train.tar -C ILSVRC2012/raw-data/imagenet-data/train/ && cd ILSVRC2012/raw-data/imagenet-data/train
find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done
cd -

# 执行转换
mkdir -p ILSVRC2012/output
python build_imagenet_data.py --train_directory=ILSVRC2012/raw-data/imagenet-data/train --validation_directory=ILSVRC2012/raw-data/imagenet-data/validation --output_directory=ILSVRC2012/output --imagenet_metadata_file=imagenet_metadata.txt --labels_file=imagenet_lsvrc_2015_synsets.txt
```

上面步骤执行完后，路径ILSVRC2012/output包含128个validation开头的验证集文件和1024个train开头的训练集文件。需要分别将验证集和数据集移动到共享目录/userhome/datasets/imagenet对应路径下

```
mkdir -p /userhome/datasets/imagenet/train
mkdir -p /userhome/datasets/imagenet/val
mv ILSVRC2012/output/train-* /userhome/datasets/imagenet/train
mv ILSVRC2012/output/validation-* /userhome/datasets/imagenet/val
```