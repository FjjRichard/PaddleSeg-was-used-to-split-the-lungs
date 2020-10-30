# PaddleSeg-was-used-to-split-the-lungs
PaddleSeg was used to divide the chest lungs with a chest X-ray

**项目背景**

欢迎百度aistudio中叉项目并运行。https://aistudio.baidu.com/aistudio/projectdetail/1170556?shared=1

参加了aistudio的分割课程之后【课程链接[ https://aistudio.baidu.com/aistudio/course/introduce/1767 ]（HTTP：//）），试效果，就使用了PaddleSeg开发套件进行自己的项目。

本来打算做胸部x光气胸的分割的。发现标记气胸的太耗时间。今天临时标记了51张肺部分割的数据。（标记了几个小时。。。。绝对原创数据。。。）试试PaddleSeg的效果。还不错
> 用**Labelme**工具标记的
![](https://ai-studio-static-online.cdn.bcebos.com/19eb62d6bf9e48d39215b70c16e3cb31ce80bb7113304fab9f896fb9a19e214e)

> 再用PaddleSeg  目录下 work/PaddleSeg/pdseg/tools/labelme2seg.py  脚本将标记好的json文件转换成Mask图片。很方便
标记效果如下

![](https://ai-studio-static-online.cdn.bcebos.com/4035e3fe788f406396b49f103f9eafbbcf37ebdb0a2c4533b8b43e85f055b1b6)



# 1、解压PaddleSeg包


```python
!unzip -q -o /home/aistudio/work/PaddleSeg.zip -d /home/aistudio/work/
```

# 2、解压数据


```python
!unzip -q -o /home/aistudio/data/data57558/chest.zip -d /home/aistudio/work/PaddleSeg/dataset
```

# 3、安装paddleSeg所需要的包


```python
%cd /home/aistudio/work/
```

    /home/aistudio/work



```python
!pip install -r /home/aistudio/work/PaddleSeg/requirements.txt
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (1.21.0)
    Requirement already satisfied: yapf==0.26.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/requirements.txt (line 2)) (0.26.0)
    Requirement already satisfied: flake8 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/requirements.txt (line 3)) (3.8.2)
    Requirement already satisfied: pyyaml>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/requirements.txt (line 4)) (5.1.2)
    Requirement already satisfied: visualdl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (2.0.3)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (0.23)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (1.3.4)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (1.3.0)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (1.15.0)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (1.4.10)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (0.10.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (16.7.9)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (2.0.1)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 3)) (2.6.0)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 3)) (0.6.1)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 3)) (2.2.0)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (7.1.2)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (2.22.0)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (1.0.0)
    Requirement already satisfied: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (1.1.1)
    Requirement already satisfied: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (3.12.2)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (1.16.4)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (0.6.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (1.25.6)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (3.0.4)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (2019.3)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (2.8.0)
    Requirement already satisfied: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (2.10.1)
    Requirement already satisfied: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (0.16.0)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (1.1.0)
    Requirement already satisfied: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (7.0)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from protobuf>=3.11.0->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (41.4.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 1)) (7.2.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl>=2.0.0->-r /home/aistudio/work/PaddleSeg/requirements.txt (line 5)) (1.1.1)


# 4、生成txt文档用于训练和验证
每一行都是原始图片 加空格  mask图片
> /home/aistudio/work/PaddleSeg/dataset/chest/origin/116.jpg /home/aistudio/work/PaddleSeg/dataset/chest/seg/116.png


```python
import os

path_origin = '/home/aistudio/work/PaddleSeg/dataset/chest/origin/'
path_seg = '/home/aistudio/work/PaddleSeg/dataset/chest/seg/'
pic_dir = os.listdir(path_origin)

f_train = open('/home/aistudio/work/PaddleSeg/dataset/chest/train_list.txt', 'w')
f_val = open('/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt', 'w')

for i in range(len(pic_dir)):
    if i % 10 != 0:
        f_train.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')
    else:
        f_val.write(path_origin + pic_dir[i] + ' ' + path_seg + pic_dir[i].split('.')[0] + '.png' + '\n')

f_train.close()
f_val.close()
```

```
#文档配置如下

# 数据集配置
DATASET:
    DATA_DIR: "/home/aistudio/work/PaddleSeg/dataset/chest/"
    NUM_CLASSES: 2
    TEST_FILE_LIST: "/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt"
    TRAIN_FILE_LIST: "/home/aistudio/work/PaddleSeg/dataset/chest/train_list.txt"
    VAL_FILE_LIST: "/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt"
    VIS_FILE_LIST: "/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt"

# 预训练模型配置
MODEL:
    MODEL_NAME: "unet"
    DEFAULT_NORM_TYPE: "bn"

# 其他配置
TRAIN_CROP_SIZE: (512, 512)
EVAL_CROP_SIZE: (512, 512)
AUG:
    AUG_METHOD: "unpadding"
    FIX_RESIZE_SIZE: (512, 512)
    # 图像镜像左右翻转
    MIRROR: True
    RICH_CROP:
        # RichCrop数据增广开关，用于提升模型鲁棒性
        ENABLE: True
        # 图像旋转最大角度，0-90
        MAX_ROTATION: 15
        # 裁取图像与原始图像面积比，0-1
        MIN_AREA_RATIO: 0.5
        # 裁取图像宽高比范围，非负
        ASPECT_RATIO: 0.33
        # 亮度调节范围，0-1
        BRIGHTNESS_JITTER_RATIO: 0.2
        # 饱和度调节范围，0-1
        SATURATION_JITTER_RATIO: 0.2
        # 对比度调节范围，0-1
        CONTRAST_JITTER_RATIO: 0.2
        # 图像模糊开关，True/False
        BLUR: False
        # 图像启动模糊百分比，0-1
        BLUR_RATIO: 0.1
BATCH_SIZE: 4
TRAIN:
    PRETRAINED_MODEL_DIR: "/home/aistudio/work/PaddleSeg/pretrained_model/unet_bn_coco/"
    MODEL_SAVE_DIR: "/home/aistudio/work/saved_model/unet_optic/"
    SNAPSHOT_EPOCH: 5
TEST:
    TEST_MODEL: "/home/aistudio/work/saved_model/unet_optic/final"
SOLVER:
    NUM_EPOCHS: 500
    LR: 0.001
    LR_POLICY: "poly"
    OPTIMIZER: "adam"

```

# 5、开始训练
先下载预训练模型

> !python /home/aistudio/work/PaddleSeg/pretrained_model/download_model.py "unet_bn_coco"
开始训练

> !python /home/aistudio/work/PaddleSeg/pdseg/train.py --use_gpu --cfg /home/aistudio/work/PaddleSeg/configs/unet_optic.yaml --do_eval 
--do_eval  
为了保存模型的时候就验证一次



```python
#下载预训练模型
!python /home/aistudio/work/PaddleSeg/pretrained_model/download_model.py "unet_bn_coco"
```

    Downloading unet_coco_v3.tgz
    [==================================================] 100.00%
    Uncompress unet_coco_v3.tgz
    [==================================================] 100.00%
    Pretrained Model download success!



```python
# Training
!export CUDA_VISIBLE_DEVICES=0
# 训练
!python /home/aistudio/work/PaddleSeg/pdseg/train.py --use_gpu --cfg /home/aistudio/work/PaddleSeg/configs/unet_optic.yaml --do_eval 
```

    Save model checkpoint to /home/aistudio/work/saved_model/unet_optic/final

# 6、进行验证


```python
# Evaluation
!python /home/aistudio/work/PaddleSeg/pdseg/eval.py --cfg /home/aistudio/work/PaddleSeg/configs/unet_optic.yaml \
                        --use_gpu \
                        EVAL_CROP_SIZE "(512, 512)"
```

    {'AUG': {'AUG_METHOD': 'unpadding',
             'FIX_RESIZE_SIZE': (512, 512),
             'FLIP': False,
             'FLIP_RATIO': 0.5,
             'INF_RESIZE_VALUE': 500,
             'MAX_RESIZE_VALUE': 600,
             'MAX_SCALE_FACTOR': 2.0,
             'MIN_RESIZE_VALUE': 400,
             'MIN_SCALE_FACTOR': 0.5,
             'MIRROR': True,
             'RICH_CROP': {'ASPECT_RATIO': 0.33,
                           'BLUR': False,
                           'BLUR_RATIO': 0.1,
                           'BRIGHTNESS_JITTER_RATIO': 0.2,
                           'CONTRAST_JITTER_RATIO': 0.2,
                           'ENABLE': True,
                           'MAX_ROTATION': 15,
                           'MIN_AREA_RATIO': 0.5,
                           'SATURATION_JITTER_RATIO': 0.2},
             'SCALE_STEP_SIZE': 0.25,
             'TO_RGB': False},
     'BATCH_SIZE': 4,
     'DATALOADER': {'BUF_SIZE': 256, 'NUM_WORKERS': 8},
     'DATASET': {'DATA_DIM': 3,
                 'DATA_DIR': '/home/aistudio/work/PaddleSeg/dataset/chest/',
                 'IGNORE_INDEX': 255,
                 'IMAGE_TYPE': 'rgb',
                 'NUM_CLASSES': 2,
                 'PADDING_VALUE': [127.5, 127.5, 127.5],
                 'SEPARATOR': ' ',
                 'TEST_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt',
                 'TEST_TOTAL_IMAGES': 6,
                 'TRAIN_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/train_list.txt',
                 'TRAIN_TOTAL_IMAGES': 45,
                 'VAL_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt',
                 'VAL_TOTAL_IMAGES': 6,
                 'VIS_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt'},
     'EVAL_CROP_SIZE': (512, 512),
     'FREEZE': {'MODEL_FILENAME': '__model__',
                'PARAMS_FILENAME': '__params__',
                'SAVE_DIR': 'freeze_model'},
     'MEAN': [0.5, 0.5, 0.5],
     'MODEL': {'BN_MOMENTUM': 0.99,
               'DEEPLAB': {'ASPP_WITH_SEP_CONV': True,
                           'BACKBONE': 'xception_65',
                           'BACKBONE_LR_MULT_LIST': None,
                           'DECODER': {'CONV_FILTERS': 256,
                                       'OUTPUT_IS_LOGITS': False,
                                       'USE_SUM_MERGE': False},
                           'DECODER_USE_SEP_CONV': True,
                           'DEPTH_MULTIPLIER': 1.0,
                           'ENABLE_DECODER': True,
                           'ENCODER': {'ADD_IMAGE_LEVEL_FEATURE': True,
                                       'ASPP_CONVS_FILTERS': 256,
                                       'ASPP_RATIOS': None,
                                       'ASPP_WITH_CONCAT_PROJECTION': True,
                                       'ASPP_WITH_SE': False,
                                       'POOLING_CROP_SIZE': None,
                                       'POOLING_STRIDE': [1, 1],
                                       'SE_USE_QSIGMOID': False},
                           'ENCODER_WITH_ASPP': True,
                           'OUTPUT_STRIDE': 16},
               'DEFAULT_EPSILON': 1e-05,
               'DEFAULT_GROUP_NUMBER': 32,
               'DEFAULT_NORM_TYPE': 'bn',
               'FP16': False,
               'HRNET': {'STAGE2': {'NUM_CHANNELS': [40, 80], 'NUM_MODULES': 1},
                         'STAGE3': {'NUM_CHANNELS': [40, 80, 160],
                                    'NUM_MODULES': 4},
                         'STAGE4': {'NUM_CHANNELS': [40, 80, 160, 320],
                                    'NUM_MODULES': 3}},
               'ICNET': {'DEPTH_MULTIPLIER': 0.5, 'LAYERS': 50},
               'MODEL_NAME': 'unet',
               'MULTI_LOSS_WEIGHT': [1.0],
               'OCR': {'OCR_KEY_CHANNELS': 256, 'OCR_MID_CHANNELS': 512},
               'PSPNET': {'DEPTH_MULTIPLIER': 1, 'LAYERS': 50},
               'SCALE_LOSS': 'DYNAMIC',
               'UNET': {'UPSAMPLE_MODE': 'bilinear'}},
     'NUM_TRAINERS': 1,
     'SLIM': {'KNOWLEDGE_DISTILL': False,
              'KNOWLEDGE_DISTILL_IS_TEACHER': False,
              'KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR': '',
              'NAS_ADDRESS': '',
              'NAS_IS_SERVER': True,
              'NAS_PORT': 23333,
              'NAS_SEARCH_STEPS': 100,
              'NAS_SPACE_NAME': '',
              'NAS_START_EVAL_EPOCH': 0,
              'PREPROCESS': False,
              'PRUNE_PARAMS': '',
              'PRUNE_RATIOS': []},
     'SOLVER': {'BEGIN_EPOCH': 1,
                'CROSS_ENTROPY_WEIGHT': None,
                'DECAY_EPOCH': [10, 20],
                'GAMMA': 0.1,
                'LOSS': ['softmax_loss'],
                'LOSS_WEIGHT': {'BCE_LOSS': 1,
                                'DICE_LOSS': 1,
                                'LOVASZ_HINGE_LOSS': 1,
                                'LOVASZ_SOFTMAX_LOSS': 1,
                                'SOFTMAX_LOSS': 1},
                'LR': 0.001,
                'LR_POLICY': 'poly',
                'LR_WARMUP': False,
                'LR_WARMUP_STEPS': 2000,
                'MOMENTUM': 0.9,
                'MOMENTUM2': 0.999,
                'NUM_EPOCHS': 500,
                'OPTIMIZER': 'adam',
                'POWER': 0.9,
                'WEIGHT_DECAY': 4e-05},
     'STD': [0.5, 0.5, 0.5],
     'TEST': {'TEST_MODEL': '/home/aistudio/work/saved_model/unet_optic/final'},
     'TRAIN': {'MODEL_SAVE_DIR': '/home/aistudio/work/saved_model/unet_optic/',
               'PRETRAINED_MODEL_DIR': '/home/aistudio/work/PaddleSeg/pretrained_model/unet_bn_coco/',
               'RESUME_MODEL_DIR': '',
               'SNAPSHOT_EPOCH': 5,
               'SYNC_BATCH_NORM': False},
     'TRAINER_ID': 0,
     'TRAIN_CROP_SIZE': (512, 512)}
    #Device count: 1
    W1029 21:27:02.416762  7698 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 9.0
    W1029 21:27:02.422560  7698 device_context.cc:260] device: 0, cuDNN Version: 7.6.
    load test model: /home/aistudio/work/saved_model/unet_optic/final
    [EVAL]step=1 loss=0.01955 acc=0.9926 IoU=0.9653 step/sec=2.12 | ETA 00:00:00
    [EVAL]step=2 loss=0.01446 acc=0.9931 IoU=0.9660 step/sec=13.80 | ETA 00:00:00
    [EVAL]#image=6 acc=0.9931 IoU=0.9660
    [EVAL]Category IoU: [0.9922 0.9398]
    [EVAL]Category Acc: [0.9955 0.9732]
    [EVAL]Kappa:0.9650


# 7、进行结果可视化


```python
# Visualization
!python /home/aistudio/work/PaddleSeg/pdseg/vis.py  --cfg /home/aistudio/work/PaddleSeg/configs/unet_optic.yaml \
                        --vis_dir visual/unet_pet \
                        --use_gpu 
```

    {'AUG': {'AUG_METHOD': 'unpadding',
             'FIX_RESIZE_SIZE': (512, 512),
             'FLIP': False,
             'FLIP_RATIO': 0.5,
             'INF_RESIZE_VALUE': 500,
             'MAX_RESIZE_VALUE': 600,
             'MAX_SCALE_FACTOR': 2.0,
             'MIN_RESIZE_VALUE': 400,
             'MIN_SCALE_FACTOR': 0.5,
             'MIRROR': True,
             'RICH_CROP': {'ASPECT_RATIO': 0.33,
                           'BLUR': False,
                           'BLUR_RATIO': 0.1,
                           'BRIGHTNESS_JITTER_RATIO': 0.2,
                           'CONTRAST_JITTER_RATIO': 0.2,
                           'ENABLE': True,
                           'MAX_ROTATION': 15,
                           'MIN_AREA_RATIO': 0.5,
                           'SATURATION_JITTER_RATIO': 0.2},
             'SCALE_STEP_SIZE': 0.25,
             'TO_RGB': False},
     'BATCH_SIZE': 4,
     'DATALOADER': {'BUF_SIZE': 256, 'NUM_WORKERS': 8},
     'DATASET': {'DATA_DIM': 3,
                 'DATA_DIR': '/home/aistudio/work/PaddleSeg/dataset/chest/',
                 'IGNORE_INDEX': 255,
                 'IMAGE_TYPE': 'rgb',
                 'NUM_CLASSES': 2,
                 'PADDING_VALUE': [127.5, 127.5, 127.5],
                 'SEPARATOR': ' ',
                 'TEST_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt',
                 'TEST_TOTAL_IMAGES': 6,
                 'TRAIN_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/train_list.txt',
                 'TRAIN_TOTAL_IMAGES': 45,
                 'VAL_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt',
                 'VAL_TOTAL_IMAGES': 6,
                 'VIS_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt'},
     'EVAL_CROP_SIZE': (512, 512),
     'FREEZE': {'MODEL_FILENAME': '__model__',
                'PARAMS_FILENAME': '__params__',
                'SAVE_DIR': 'freeze_model'},
     'MEAN': [0.5, 0.5, 0.5],
     'MODEL': {'BN_MOMENTUM': 0.99,
               'DEEPLAB': {'ASPP_WITH_SEP_CONV': True,
                           'BACKBONE': 'xception_65',
                           'BACKBONE_LR_MULT_LIST': None,
                           'DECODER': {'CONV_FILTERS': 256,
                                       'OUTPUT_IS_LOGITS': False,
                                       'USE_SUM_MERGE': False},
                           'DECODER_USE_SEP_CONV': True,
                           'DEPTH_MULTIPLIER': 1.0,
                           'ENABLE_DECODER': True,
                           'ENCODER': {'ADD_IMAGE_LEVEL_FEATURE': True,
                                       'ASPP_CONVS_FILTERS': 256,
                                       'ASPP_RATIOS': None,
                                       'ASPP_WITH_CONCAT_PROJECTION': True,
                                       'ASPP_WITH_SE': False,
                                       'POOLING_CROP_SIZE': None,
                                       'POOLING_STRIDE': [1, 1],
                                       'SE_USE_QSIGMOID': False},
                           'ENCODER_WITH_ASPP': True,
                           'OUTPUT_STRIDE': 16},
               'DEFAULT_EPSILON': 1e-05,
               'DEFAULT_GROUP_NUMBER': 32,
               'DEFAULT_NORM_TYPE': 'bn',
               'FP16': False,
               'HRNET': {'STAGE2': {'NUM_CHANNELS': [40, 80], 'NUM_MODULES': 1},
                         'STAGE3': {'NUM_CHANNELS': [40, 80, 160],
                                    'NUM_MODULES': 4},
                         'STAGE4': {'NUM_CHANNELS': [40, 80, 160, 320],
                                    'NUM_MODULES': 3}},
               'ICNET': {'DEPTH_MULTIPLIER': 0.5, 'LAYERS': 50},
               'MODEL_NAME': 'unet',
               'MULTI_LOSS_WEIGHT': [1.0],
               'OCR': {'OCR_KEY_CHANNELS': 256, 'OCR_MID_CHANNELS': 512},
               'PSPNET': {'DEPTH_MULTIPLIER': 1, 'LAYERS': 50},
               'SCALE_LOSS': 'DYNAMIC',
               'UNET': {'UPSAMPLE_MODE': 'bilinear'}},
     'NUM_TRAINERS': 1,
     'SLIM': {'KNOWLEDGE_DISTILL': False,
              'KNOWLEDGE_DISTILL_IS_TEACHER': False,
              'KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR': '',
              'NAS_ADDRESS': '',
              'NAS_IS_SERVER': True,
              'NAS_PORT': 23333,
              'NAS_SEARCH_STEPS': 100,
              'NAS_SPACE_NAME': '',
              'NAS_START_EVAL_EPOCH': 0,
              'PREPROCESS': False,
              'PRUNE_PARAMS': '',
              'PRUNE_RATIOS': []},
     'SOLVER': {'BEGIN_EPOCH': 1,
                'CROSS_ENTROPY_WEIGHT': None,
                'DECAY_EPOCH': [10, 20],
                'GAMMA': 0.1,
                'LOSS': ['softmax_loss'],
                'LOSS_WEIGHT': {'BCE_LOSS': 1,
                                'DICE_LOSS': 1,
                                'LOVASZ_HINGE_LOSS': 1,
                                'LOVASZ_SOFTMAX_LOSS': 1,
                                'SOFTMAX_LOSS': 1},
                'LR': 0.001,
                'LR_POLICY': 'poly',
                'LR_WARMUP': False,
                'LR_WARMUP_STEPS': 2000,
                'MOMENTUM': 0.9,
                'MOMENTUM2': 0.999,
                'NUM_EPOCHS': 500,
                'OPTIMIZER': 'adam',
                'POWER': 0.9,
                'WEIGHT_DECAY': 4e-05},
     'STD': [0.5, 0.5, 0.5],
     'TEST': {'TEST_MODEL': '/home/aistudio/work/saved_model/unet_optic/final'},
     'TRAIN': {'MODEL_SAVE_DIR': '/home/aistudio/work/saved_model/unet_optic/',
               'PRETRAINED_MODEL_DIR': '/home/aistudio/work/PaddleSeg/pretrained_model/unet_bn_coco/',
               'RESUME_MODEL_DIR': '',
               'SNAPSHOT_EPOCH': 5,
               'SYNC_BATCH_NORM': False},
     'TRAINER_ID': 0,
     'TRAIN_CROP_SIZE': (512, 512)}
    W1029 21:27:51.493047  7853 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 11.0, Runtime API Version: 9.0
    W1029 21:27:51.498473  7853 device_context.cc:260] device: 0, cuDNN Version: 7.6.
    load test model: /home/aistudio/work/saved_model/unet_optic/final
    #1 visualize image path: visual/unet_pet/11.png
    #2 visualize image path: visual/unet_pet/169.png
    #3 visualize image path: visual/unet_pet/164.png
    #4 visualize image path: visual/unet_pet/119.png
    #5 visualize image path: visual/unet_pet/13.png
    #6 visualize image path: visual/unet_pet/109.png


# 500轮后，效果还不错


```python
import matplotlib.pyplot as plt

# 定义显示函数
def display(img_dir):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask','Predicted Mask']
    
    for i in range(len(title)):
        plt.subplot(1, len(img_dir), i+1)
        plt.title(title[i])
        img = plt.imread(img_dir[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()

# 显示分割效果
# 注：这里仅显示其中一张图片的效果。    
image_dir = "/home/aistudio/work/PaddleSeg/dataset/chest/origin/109.jpg"
true_dir = '/home/aistudio/work/PaddleSeg/dataset/chest/seg/109.png'
mask_dir = "/home/aistudio/work/visual/unet_pet/109.png"
imgs = [image_dir, true_dir,mask_dir]
display(imgs)
```


![png](output_19_0.png)


# 8、 模型导出

导出位置/home/aistudio/freeze_model


```python
!python /home/aistudio/work/PaddleSeg/pdseg/export_model.py --cfg /home/aistudio/work/PaddleSeg/configs/unet_optic.yaml 
```

    {'AUG': {'AUG_METHOD': 'unpadding',
             'FIX_RESIZE_SIZE': (512, 512),
             'FLIP': False,
             'FLIP_RATIO': 0.5,
             'INF_RESIZE_VALUE': 500,
             'MAX_RESIZE_VALUE': 600,
             'MAX_SCALE_FACTOR': 2.0,
             'MIN_RESIZE_VALUE': 400,
             'MIN_SCALE_FACTOR': 0.5,
             'MIRROR': True,
             'RICH_CROP': {'ASPECT_RATIO': 0.33,
                           'BLUR': False,
                           'BLUR_RATIO': 0.1,
                           'BRIGHTNESS_JITTER_RATIO': 0.2,
                           'CONTRAST_JITTER_RATIO': 0.2,
                           'ENABLE': True,
                           'MAX_ROTATION': 15,
                           'MIN_AREA_RATIO': 0.5,
                           'SATURATION_JITTER_RATIO': 0.2},
             'SCALE_STEP_SIZE': 0.25,
             'TO_RGB': False},
     'BATCH_SIZE': 4,
     'DATALOADER': {'BUF_SIZE': 256, 'NUM_WORKERS': 8},
     'DATASET': {'DATA_DIM': 3,
                 'DATA_DIR': '/home/aistudio/work/PaddleSeg/dataset/chest/',
                 'IGNORE_INDEX': 255,
                 'IMAGE_TYPE': 'rgb',
                 'NUM_CLASSES': 2,
                 'PADDING_VALUE': [127.5, 127.5, 127.5],
                 'SEPARATOR': ' ',
                 'TEST_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt',
                 'TEST_TOTAL_IMAGES': 6,
                 'TRAIN_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/train_list.txt',
                 'TRAIN_TOTAL_IMAGES': 45,
                 'VAL_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt',
                 'VAL_TOTAL_IMAGES': 6,
                 'VIS_FILE_LIST': '/home/aistudio/work/PaddleSeg/dataset/chest/val_list.txt'},
     'EVAL_CROP_SIZE': (512, 512),
     'FREEZE': {'MODEL_FILENAME': '__model__',
                'PARAMS_FILENAME': '__params__',
                'SAVE_DIR': 'freeze_model'},
     'MEAN': [0.5, 0.5, 0.5],
     'MODEL': {'BN_MOMENTUM': 0.99,
               'DEEPLAB': {'ASPP_WITH_SEP_CONV': True,
                           'BACKBONE': 'xception_65',
                           'BACKBONE_LR_MULT_LIST': None,
                           'DECODER': {'CONV_FILTERS': 256,
                                       'OUTPUT_IS_LOGITS': False,
                                       'USE_SUM_MERGE': False},
                           'DECODER_USE_SEP_CONV': True,
                           'DEPTH_MULTIPLIER': 1.0,
                           'ENABLE_DECODER': True,
                           'ENCODER': {'ADD_IMAGE_LEVEL_FEATURE': True,
                                       'ASPP_CONVS_FILTERS': 256,
                                       'ASPP_RATIOS': None,
                                       'ASPP_WITH_CONCAT_PROJECTION': True,
                                       'ASPP_WITH_SE': False,
                                       'POOLING_CROP_SIZE': None,
                                       'POOLING_STRIDE': [1, 1],
                                       'SE_USE_QSIGMOID': False},
                           'ENCODER_WITH_ASPP': True,
                           'OUTPUT_STRIDE': 16},
               'DEFAULT_EPSILON': 1e-05,
               'DEFAULT_GROUP_NUMBER': 32,
               'DEFAULT_NORM_TYPE': 'bn',
               'FP16': False,
               'HRNET': {'STAGE2': {'NUM_CHANNELS': [40, 80], 'NUM_MODULES': 1},
                         'STAGE3': {'NUM_CHANNELS': [40, 80, 160],
                                    'NUM_MODULES': 4},
                         'STAGE4': {'NUM_CHANNELS': [40, 80, 160, 320],
                                    'NUM_MODULES': 3}},
               'ICNET': {'DEPTH_MULTIPLIER': 0.5, 'LAYERS': 50},
               'MODEL_NAME': 'unet',
               'MULTI_LOSS_WEIGHT': [1.0],
               'OCR': {'OCR_KEY_CHANNELS': 256, 'OCR_MID_CHANNELS': 512},
               'PSPNET': {'DEPTH_MULTIPLIER': 1, 'LAYERS': 50},
               'SCALE_LOSS': 'DYNAMIC',
               'UNET': {'UPSAMPLE_MODE': 'bilinear'}},
     'NUM_TRAINERS': 1,
     'SLIM': {'KNOWLEDGE_DISTILL': False,
              'KNOWLEDGE_DISTILL_IS_TEACHER': False,
              'KNOWLEDGE_DISTILL_TEACHER_MODEL_DIR': '',
              'NAS_ADDRESS': '',
              'NAS_IS_SERVER': True,
              'NAS_PORT': 23333,
              'NAS_SEARCH_STEPS': 100,
              'NAS_SPACE_NAME': '',
              'NAS_START_EVAL_EPOCH': 0,
              'PREPROCESS': False,
              'PRUNE_PARAMS': '',
              'PRUNE_RATIOS': []},
     'SOLVER': {'BEGIN_EPOCH': 1,
                'CROSS_ENTROPY_WEIGHT': None,
                'DECAY_EPOCH': [10, 20],
                'GAMMA': 0.1,
                'LOSS': ['softmax_loss'],
                'LOSS_WEIGHT': {'BCE_LOSS': 1,
                                'DICE_LOSS': 1,
                                'LOVASZ_HINGE_LOSS': 1,
                                'LOVASZ_SOFTMAX_LOSS': 1,
                                'SOFTMAX_LOSS': 1},
                'LR': 0.001,
                'LR_POLICY': 'poly',
                'LR_WARMUP': False,
                'LR_WARMUP_STEPS': 2000,
                'MOMENTUM': 0.9,
                'MOMENTUM2': 0.999,
                'NUM_EPOCHS': 500,
                'OPTIMIZER': 'adam',
                'POWER': 0.9,
                'WEIGHT_DECAY': 4e-05},
     'STD': [0.5, 0.5, 0.5],
     'TEST': {'TEST_MODEL': '/home/aistudio/work/saved_model/unet_optic/final/'},
     'TRAIN': {'MODEL_SAVE_DIR': '/home/aistudio/work/saved_model/unet_optic/',
               'PRETRAINED_MODEL_DIR': '/home/aistudio/work/PaddleSeg/pretrained_model/unet_bn_coco/',
               'RESUME_MODEL_DIR': '',
               'SNAPSHOT_EPOCH': 5,
               'SYNC_BATCH_NORM': False},
     'TRAINER_ID': 0,
     'TRAIN_CROP_SIZE': (512, 512)}
    Exporting inference model...
    load test model: /home/aistudio/work/saved_model/unet_optic/final/
    Inference model exported!
    Exporting inference model config...
    Inference model saved : [freeze_model/deploy.yaml]


# 通过导出模型进行  新数据预测效果


```python
#预测前要安装 某些依赖包
!pip install -r /home/aistudio/work/PaddleSeg/deploy/python/requirements.txt
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting python-gflags (from -r /home/aistudio/work/PaddleSeg/deploy/python/requirements.txt (line 1))
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/df/ec/e31302d355bcb9d207d9b858adc1ecc4a6d8c855730c8ba4ddbdd3f8eb8d/python-gflags-3.1.2.tar.gz (52kB)
    [K     |████████████████████████████████| 61kB 14.9MB/s eta 0:00:01
    [?25hRequirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/deploy/python/requirements.txt (line 2)) (5.1.2)
    Requirement already satisfied: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/deploy/python/requirements.txt (line 3)) (1.16.4)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from -r /home/aistudio/work/PaddleSeg/deploy/python/requirements.txt (line 4)) (4.1.1.26)
    Collecting futures (from -r /home/aistudio/work/PaddleSeg/deploy/python/requirements.txt (line 5))
      Downloading https://mirror.baidu.com/pypi/packages/05/80/f41cca0ea1ff69bce7e7a7d76182b47bb4e1a494380a532af3e8ee70b9ec/futures-3.1.1-py3-none-any.whl
    Building wheels for collected packages: python-gflags
      Building wheel for python-gflags (setup.py) ... [?25ldone
    [?25h  Created wheel for python-gflags: filename=python_gflags-3.1.2-cp37-none-any.whl size=57366 sha256=5cfcd231fb215e0626cf90cee034e6ffbb72627a524c7cc8eac9f83d984cea8d
      Stored in directory: /home/aistudio/.cache/pip/wheels/77/80/3c/8ec1509c7aa89b0911b46d83d503412fabd6635d68c79e5c06
    Successfully built python-gflags
    Installing collected packages: python-gflags, futures
    Successfully installed futures-3.1.1 python-gflags-3.1.2



```python
!python /home/aistudio/work/PaddleSeg/deploy/python/infer.py --conf=/home/aistudio/freeze_model/deploy.yaml --input_dir=/home/aistudio/work/test_img
```

    I1030 08:04:05.713804  1791 analysis_predictor.cc:138] Profiler is deactivated, and no profiling report will be generated.
    I1030 08:04:05.965926  1791 analysis_predictor.cc:875] MODEL VERSION: 1.8.4
    I1030 08:04:05.965970  1791 analysis_predictor.cc:877] PREDICTOR VERSION: 1.8.4
    [1m[35m--- Running analysis [ir_graph_build_pass][0m
    [1m[35m--- Running analysis [ir_graph_clean_pass][0m
    [1m[35m--- Running analysis [ir_analysis_pass][0m
    [32m--- Running IR pass [is_test_pass][0m
    [32m--- Running IR pass [simplify_with_basic_ops_pass][0m
    [32m--- Running IR pass [conv_affine_channel_fuse_pass][0m
    [32m--- Running IR pass [conv_eltwiseadd_affine_channel_fuse_pass][0m
    [32m--- Running IR pass [conv_bn_fuse_pass][0m
    I1030 08:04:06.044466  1791 graph_pattern_detector.cc:101] ---  detected 18 subgraphs
    [32m--- Running IR pass [conv_eltwiseadd_bn_fuse_pass][0m
    [32m--- Running IR pass [embedding_eltwise_layernorm_fuse_pass][0m
    [32m--- Running IR pass [multihead_matmul_fuse_pass_v2][0m
    [32m--- Running IR pass [fc_fuse_pass][0m
    [32m--- Running IR pass [fc_elementwise_layernorm_fuse_pass][0m
    [32m--- Running IR pass [conv_elementwise_add_act_fuse_pass][0m
    I1030 08:04:06.061342  1791 graph_pattern_detector.cc:101] ---  detected 18 subgraphs
    [32m--- Running IR pass [conv_elementwise_add2_act_fuse_pass][0m
    [32m--- Running IR pass [conv_elementwise_add_fuse_pass][0m
    [32m--- Running IR pass [transpose_flatten_concat_fuse_pass][0m
    [32m--- Running IR pass [runtime_context_cache_pass][0m
    [1m[35m--- Running analysis [ir_params_sync_among_devices_pass][0m
    I1030 08:04:06.065783  1791 ir_params_sync_among_devices_pass.cc:41] Sync params from CPU to GPU
    [1m[35m--- Running analysis [adjust_cudnn_workspace_size_pass][0m
    [1m[35m--- Running analysis [inference_op_replace_pass][0m
    [1m[35m--- Running analysis [memory_optimize_pass][0m
    I1030 08:04:06.103349  1791 memory_optimize_pass.cc:223] Cluster name : relu_7.tmp_0  size: 8388608
    I1030 08:04:06.103375  1791 memory_optimize_pass.cc:223] Cluster name : relu_9.tmp_0  size: 2097152
    I1030 08:04:06.103379  1791 memory_optimize_pass.cc:223] Cluster name : bilinear_interp_1.tmp_0  size: 16777216
    I1030 08:04:06.103381  1791 memory_optimize_pass.cc:223] Cluster name : relu_3.tmp_0  size: 33554432
    I1030 08:04:06.103384  1791 memory_optimize_pass.cc:223] Cluster name : relu_1.tmp_0  size: 67108864
    I1030 08:04:06.103404  1791 memory_optimize_pass.cc:223] Cluster name : image  size: 3145728
    I1030 08:04:06.103410  1791 memory_optimize_pass.cc:223] Cluster name : relu_16.tmp_0  size: 67108864
    I1030 08:04:06.103413  1791 memory_optimize_pass.cc:223] Cluster name : concat_3.tmp_0  size: 134217728
    [1m[35m--- Running analysis [ir_graph_to_program_pass][0m
    I1030 08:04:06.111246  1791 analysis_predictor.cc:496] ======= optimize end =======
    W1030 08:04:06.169015  1791 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 10.1, Runtime API Version: 9.0
    W1030 08:04:06.708915  1791 device_context.cc:260] device: 0, cuDNN Version: 7.6.
    save result of [/home/aistudio/work/test_img/272.jpg] done.
    save result of [/home/aistudio/work/test_img/325.jpg] done.
    save result of [/home/aistudio/work/test_img/271.jpg] done.
    images_num=[3],preprocessing_time=[0.110725],infer_time=[8.537018],postprocessing_time=[0.748331],total_runtime=[9.397624]


# 数据预测结果可视化


```python
#把mask和原图融合在一起
import cv2
import os
def visualize(image, result, save_dir=None, weight=0.6):
    # color_map = get_color_map_list(256)
    # color_map = np.array(color_map).astype("uint8")
    # # Use OpenCV LUT for color mapping
    # c1 = cv2.LUT(result, color_map[:, 0])
    # c2 = cv2.LUT(result, color_map[:, 1])
    # c3 = cv2.LUT(result, color_map[:, 2])
    # pseudo_img = np.dstack((c1, c2, c3))
    pseudo_img = cv2.imread(result)

    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, 1, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, 'vis_'+image_name)
        cv2.imwrite(out_path, vis_result)
        print("保存成功")
    else:
        return vis_result
#['272','271','325']图片文件名
for img_name in ['272','271','325']:
    dis_img_path = '/home/aistudio/work/test_img/' + img_name + '.jpg'
    pre_img_path = '/home/aistudio/work/test_img/' + img_name + '_jpg_result.png'
    save_result_path = '/home/aistudio/work/result'
    visualize(dis_img_path,pre_img_path,save_dir=save_result_path,weight=0.7)
```

    保存成功
    保存成功
    保存成功



```python
# 显示分割效果 
import matplotlib.pyplot as plt
# 定义显示函数
def display(img_dir,titles):
    plt.figure(figsize=(15, 18))
    title = titles
    for i in range(len(title)):
        plt.subplot(1, len(img_dir), i+1)
        plt.title(title[i])
        img = plt.imread(img_dir[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()

  
imgs_path = '/home/aistudio/work/result'
img_list = os.listdir(imgs_path)
img_list = list(img_name for img_name in img_list if 'jpg'  in img_name)
img_titles = img_list
img_list = list(os.path.join(imgs_path,img_name) for img_name in img_list )
display(img_list,img_titles)
```


![png](output_27_0.png)


# 9、进行mase图和原图合并，并用轮廓显示


```python
import cv2
import numpy as np
def union_image_mask(image_path, mask_path, num):
    # 读取原图
    image = cv2.imread(image_path)
    # 读取分割mask，这里mask  背景类是0，第一类是彩色
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 裁剪到和原图一样大小
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    #先把mask图片转换成单通道的背景色是黑色。肺部是白色的图片，用来查找轮廓
    mask_3d = np.ones((h, w), dtype='uint8')*0
    # 背景是黑色==0，伪彩是>0 所以  伪彩都转换成白色。
    mask_3d[mask_2d[:, :] > 1.0] = 255
    print('原图的shape:{}'.format(image.shape))
    print('mask的shape:{}'.format(mask_3d.shape))
    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2,lineType=cv2.LINE_AA)
    cv2.imwrite('img_profile.jpg',image)
    print("保存完成")

union_image_mask('/home/aistudio/work/test_img/325.jpg',
"/home/aistudio/work/test_img/325_jpg_result.png",1)
```

    原图的shape:(1052, 1000, 3)
    mask的shape:(1052, 1000)



```python
# 显示轮廓合并效果 
import matplotlib.pyplot as plt
img_path = '/home/aistudio/img_profile.jpg'
# 定义显示函数
plt.figure(figsize=(10, 10))
title = 'profile'
plt.title(title)
img = plt.imread(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()

```


![png](output_30_0.png)


# 10、进行mase图和原图合并，然后只保留肺部，用着其他任务会更加关注肺部


```python
import cv2
import numpy as np
def union_image_mask(image_path, mask_path, num):
    # 读取原图
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    # 读取分割mask，这里mask  背景类是0，第一类是彩色
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 裁剪到和原图一样大小
    # mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    #先把mask图片转换成单通道的背景色白色，肺部是黑色
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # 背景是黑色==0，伪彩是>0 所以  伪彩都转换成黑色。
    mask_3d[mask_2d[:, :] > 1.0] = 0
    #要保持原图和 mask维度保持一致
    print('原图的shape:{}'.format(image.shape))
    print('mask的shape:{}'.format(mask_3d.shape))
    result = cv2.add(image, mask_3d)
    cv2.imwrite('img_add.jpg',result)
    print("保存完成")

union_image_mask('/home/aistudio/work/test_img/325.jpg',
"/home/aistudio/work/test_img/325_jpg_result.png",1)
```

    原图的shape:(1052, 1000)
    mask的shape:(1052, 1000)
    保存完成



```python
# 显示轮廓合并效果 
import matplotlib.pyplot as plt
img_path = '/home/aistudio/img_add.jpg'
# 定义显示函数
plt.figure(figsize=(10, 10))
title = 'add_image'
plt.title(title)
img = plt.imread(img_path)
plt.imshow(img,cmap='Greys_r')
plt.axis('off')
plt.show()
```


![png](output_33_0.png)


# 10、总结

paddleSeg开发套件，真的好用，基本不怎么输入代码，就可以获得不错的效果。最近paddleSeg改版了，改成动态图。不过这次项目没有动态版。
本来想试个复杂的数据。自己标记胸部X光气胸的分割数据。用来测量气胸压缩比。后来想想还是算了。下次再标记吧。

paddleSeg github地址：[https://github.com/PaddlePaddle/PaddleSeg](http://)

# 11、个人介绍

广州某医院的放射科的一名放射技师。

只是一位编程爱好者

只是想把自己的爱好融入工作中

只是想让自己通过努力获取成就和快乐

欢迎更多志同道合的朋友一起玩耍~~~

我在AI Studio上获得黄金等级，点亮5个徽章，来互关呀~ [https://aistudio.baidu.com/aistudio/personalcenter/thirdview/181096](http://)
