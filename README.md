# AI in RTC 超分辨率算法性能比较挑战赛Top5方案
任务：对图像做2倍超分，算法复杂度控制在2G FLOPS之内。

本代码采用的baseline为EDSR。
# 实验效果
---

|Model  |FLOPS|#Params|Set5(PSNR)|
|---    |---  |---    |---       |
| SRCNN |52.7G|57K    | 36.66    |
| FSRCNN|6.0G |12K    | 37.00    | 
| Ours  |1.93G| -     | 37.12    |

---
# 代码目录
- code/ 存放代码
- submit/ 结果输出
- testing_data/ 测试数据
- training_data/ 训练数据
- trained_model/ 预训练模型
- tmp/ 临时文件夹

# 环境
- Python 3.6
- Pytorch 1.0.0
- numpy
- skimage
- imageio
- matplotlib
- tqdm
- CUDA版本8.0.61
- CUDNN版本8.0， V8.0.61

安装的话都直接pip安装就行了

# 运行训练代码
进入到src

cd code/src

    python main.py --model model --scale 2 --patch_size 192 --lr 1e-3 --data_range 1-3600/3601-3605 --gamma 0.2 --epochs 800 --ext sep_reset
或者

    sh train.sh

模型保存在了根目录tmp里面，一个按时间定义的文件夹下的model里面，有两个文件，model_best.pt和model_latest.pt

（注意：测试的时候要把扩展名pt改为pth,即文件名改为model.pth）

（注意：第二遍运行要去掉--ext sep_reset！！！！！！！！！！！！！！）

即运行：

    python main.py --model model --scale 2 --patch_size 192 --lr 1e-3 --data_range 1-3600/3601-3605 --gamma 0.2 --epochs 800
或

    sh train2.sh

# 运行测试代码
直接用官方的测试代码即可。
或者在code里面有一份test_src，就是官方的测试程序，改一下HR和LR的路径即可。

记得把submit里面的两个文件model.py和model.pth放到相应位置。
或者是tmp里面训好的模型（model_best.pt或者model_latest.pt，一般用best），改名成model.pth，以及code/src/model/model.py。

# 数据集
用了DIV2K和Flickr2K，需要从0001开始编号，如0001.png, 0002.png等等。
