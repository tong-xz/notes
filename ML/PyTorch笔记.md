---
typora-root-url: ./images\pytorch
---

# PyTorch 笔记

[toc]

> URL: https://www.bilibili.com/video/BV1hE411t7RN/?spm_id_from=333.337.search-card.all.click&vd_source=8bfc362662d9e1c42d3208e8dffbb371





## 1. Settings

检测本地cuda是否可用

```python
import torch
print(torch.cuda.is_avaliable())
```



## 2. Input and Tools

### 2.1 Tools

`dir()`能够知道包的结构

`help()`括号内放置特定的pytorch包，知道具体的用法



![image-20220713124134091](/D:/TyporaSpace/Images/image-20220713124134091-1683623627940-4.png)



`input = torch.randn(3, 4, 3, 3)`行数在前，列数在后。生成的矩阵如下

三个数字时候，第一个为行数

batch, , 行数，列数
$$
\begin{Bmatrix}
\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix} \\

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix} \\

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix}

\begin{bmatrix}
 1 & 1 &1 \\
 1 & 1 &1 \\
 1 & 1 &1 \\
\end{bmatrix} \\


\end{Bmatrix}
$$




#### 2.1.1 Transform

> 主要是针对Image的一些变化



##### ToTensor() 

把image信息转化为tensor信息



要先对transforms里面的工具new一个对象，然后把参数放到工具对象里面去

<img src="/D:/TyporaSpace/Images/image-20220713225549594-1683623693732-7.png" alt="image-20220713225549594" style="zoom: 67%;" />

```python
from torchvision import transforms
from PIL import Image

img_path = "data/train/ants/0013035.jpg"
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
```



##### Normalize( )

 对输入的image信息进行归一化
$$
input[channel]=\frac{input[channel]-mean[channel]}{std[channel]}
$$
因为对于图片来说总共有三个通道，所以mean和std相应的都需要输入一个list

```python
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_path = "data/train/ants/0013035.jpg"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("ToTensor", tensor_img)

print(tensor_img[0][0][0])

trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image("Normalize", img_norm)
writer.close()
```



##### Resize()

把输入的PIL image调整到参数规定的大小，输入的参数应该是 **（H, W）**

```python
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
# img是一个PIL对象
print(img_resize)
```



##### Compose()

把所有要做到变化，按照顺序以list的形式，放到Compose里面去 [transform1, transform2, tranform3, ....] 

里面的transform都是本地实例化好的工具对象， 前一个的输出会是下一个的输入

```python
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img_path = "data/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

# ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
writer.add_image("ToTensor", tensor_img)
print(tensor_img[0][0][0])

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm)

# Resize
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)

# Resize 2
trains_resize_2 = transforms.Resize(512)
# 声明好compose工具
trans_com = transforms.Compose([trains_resize_2, tensor_trans])
img_resize_2 = trans_com(img) #使用工具
writer.add_image("Resize", img_resize_2, 1)


writer.close()
```



##### RandomCrop

随机裁剪，输入(H, W)或者一个整数 S 则会裁剪一个边长为S的正方形

```python
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, tensor_trans])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Random Crop", img_crop, i) # 随机按照指定大小裁减
writer.close()
```



#### 2.1.2 Tensorboard

> 用于记录整个训练过程中loss的变化以及一些比较有用的信息

先声明一个`SummaryWriter`对象，内加参数是log的保存文件名，运行结束之后会产生一个log文件在logs路径下面，tensorboard会把所有的log文件画出来

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
writer.close()
```



`add_scalar()` 包含三个主要的参数 **tag, scalar_value, global_step** 分别对应的是下图的 **标题， y轴， x轴**

```python
add_scalar(tag, scalar_value, global_step)
	tag (string): Data identifier
	scalar_value (float or string/blobname): Value to save
	global_step (int): Global step value to record
```

![image-20220713201853591](D:\TyporaSpace\Images\image-20220713201853591.png)

**插入图片**

主要是能够看到第几步对应的图片

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "data/train/ants/0013035.jpg"
image_PIL = Image.open(image_path)
image_array = np.array(image_PIL)
# 代表的是第一步，转化成numpy之后会有特定格式，最好声明
writer.add_image("test", image_array, 1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)
    
writer.close()
```



`add_image()` 包含三个主要的参数 **tag, scalar_value, global_step** 分别对应的是下图的 **标题， image转化的tensor信息， 第几步**

`add_images()`可以加多个图片合在一起的tensor信息

```python
add_image(tag, image_tensor, global_step)
        tag (string): Data identifier
        img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
        global_step (int): Global step value to record
        walltime (float): Optional override default walltime (time.time())
          seconds after epoch of event
        dataformats (string): Image data format specification of the form
          CHW, HWC, HW, WH, etc.
```

**记得把writer close掉**

启动指令

```
tensorboard --logdir=logs --port=6006
```



#### 2.1.3 Tensor变化

##### torch.reshape()

变化tensor矩阵的形状

##### torch.flatten()

把tensor变为 1 dimension



![image-20220717151841592](/D:/TyporaSpace/Images/image-20220717151841592-1683623776005-10.png)





### 2.2 Dataset and DataLoader

#### 2.2.1 Dataset

##### I.  自定义dataset对象

**Dataset:** 本质上是提供一种方式去获取数据及其label，以及总共有多少个数据

```python
from torch.utils.data.dataset import Dataset

class MyData(Dataset):
    def __init__(self):
        # 主要是声明一些path
        
        
    def __getitem__(self):
        # 能够访问到数据集对象中的每个data
        
    def __len__(self):
        # 返回整个dataset对象的长度

```



##### II. torchvision 中常用的数据集



```python
torchvision.datasets.CIFAR10(root: str, # 声明文件夹
                             train: bool = True, #是否是train数据集
                             transform: Optional[Callable] = None, #特定的transform
                             target_transform: Optional[Callable] = None, 
                             download: bool = False)
```

可以这样打断点，debug来看数据里面到底包含了什么。可以通过`set.feature` 访问特定数据集中的特定数据，feature是数据集里面特定的variable

![image-20220715152544517](/D:/TyporaSpace/Images/image-20220715152544517-1683623800281-13.png)



<img src="E:\notes\ML\images\pytorch\image-20220715152924362.png" alt="image-20220715152924362" style="zoom:150%;" />



#### 2.2.2 DataLoader

**Dataloader:** 从dataset里面取值，并规定了如何从dataset里面取值

访问dataset[i]的时候，返回多少个值，loader会把相应的batchsize里面所有的**同类型数值**打包起来

```python
DataLoader(dataset, 
           batch_size=1, #每个batch多少个sample
           shuffle=False, # 是否随机
           sampler=None,
           batch_sampler=None, 
           num_workers=0, # 多少个进程 subprocess 去进行加载， 如果是0，默认用main 进程
           collate_fn=None,
           pin_memory=False, 
           drop_last=False, # 总sample数量 / batch大小，如果有余数，true那么就drop掉余数
           timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```



```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./CIFAR", train=False, transform=torchvision.transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


writer = SummaryWriter("logs")

step=0
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epoch {}".format(epoch), imgs, step)
        step = step+1

writer.close()
```



## 3. Neural Network

主要使用`torch.nn`，里面包含了很多不同的layer

#### 3.1 Container

写自己的神经网络的时候，一定要继承 `nn.module`

```python
class Model(nn.Module):
    def __init__(self):# 初始化函数
        super(Model, self).__init__() # 首先调用父类init函数 
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d()
        
    def forward(self, input): # 前馈神经网络
        
```



#### 3.2 Convolution Layer



```python
torch.nn.Conv2d(in_channels, # input image有多少个channel
                out_channels, 
                kernel_size, #卷积核的大小， 卷积核的数值是由参数决定的
                stride=1, 
                padding=0, # 填充多少
                dilation=1, 
                groups=1, 
                bias=True, 
                padding_mode='zeros', 
                device=None, 
                dtype=None)
```



这里input_channel=1, output channel=2, 就是卷积过程中生成两个卷积核，分别卷积得到结果

<img src="E:\notes\ML\images\pytorch\image-20220716144115658.png" alt="image-20220716144115658" style="zoom:66%;" />



```python
import torch
from torch import nn
import torchvision.datasets
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataLoader = DataLoader(dataset, 64)

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, input):
        input = self.conv1(input)
        return input


n = network()
writer = SummaryWriter("./logs")
step = 0
for data in dataLoader:
    imgs, target =  data
    output = n(imgs)

    info = torch.reshape(output, [-1, 3, 30, 30]) # -1由于从6变到3，数量未知，所以写-1. 3, channel, 30, 30图片大小
    writer.add_images('input', imgs, step)
    writer.add_images('output', info, step)

    step = step + 1
```



<img src="E:\notes\ML\images\pytorch\image-20220716154056785.png" alt="image-20220716154056785" style="zoom:50%;" />





#### 3.3 Pooling Layer

**作用：** 在[卷积神经网络](https://so.csdn.net/so/search?q=卷积神经网络&spm=1001.2101.3001.7020)中通常会在相邻的卷积层之间加入一个池化层，池化层可以有效的缩小参数矩阵的尺寸，从而减少最后连接层的中的参数数量。所以加入池化层可以加快计算速度和防止过拟合的作用。核心就是 ***减少参数量***， 从而保留特征，减少参数量



##### MaxPool2d

输入必须是`(N, C, Hin, Win)` N-batch size, C- channel

```python
torch.nn.MaxPool2d(kernel_size, 
                   stride=None, 
                   padding=0, 
                   dilation=1, # 空洞卷积会用到
                   return_indices=False, 
                   ceil_mode=False # true用ceil, false用floor, 决定了取整方式
                  )
```



![image-20220716161610297](/D:/TyporaSpace/Images/image-20220716161610297.png)

对于ceil和floor的操作，如果ceil_mode是true，即使用kernel pooling的时候，如果不全的话，取剩下的格子里面的值处理

如果ceil_mode是false，不满足kernel的大小，不处理

![image-20220716164622416](/D:/TyporaSpace/Images/image-20220716164622416-1683623849997-17.png)

```python
import torch
from torch import nn



class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool(x)
        return x

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))
n = network()
output = n(input)
print(output)
```



对实际数据集进行采样， 图片清晰度会降低，但是大致的形状还能看出来

```python
import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool(x)
        return x

step =0
n = network()
writer = SummaryWriter("./logs_maxpool")

for data in dataloader:
    imgs, targets = data
    
    writer.add_images("input", imgs, step)
    output = n(imgs)
    
    writer.add_images("output", output, step)
    step = step + 1

writer.close()
```





<img src="E:\notes\ML\images\pytorch\image-20220716232758144.png" alt="image-20220716232758144" style="zoom: 80%;" />



#### 3.4 Non-linear Activations

**作用：** 向网络中引入尽可能多的非线性特征，这样才能模拟各种曲线，以及非线性特征，不然泛化能力会不够好

##### ReLU (Rectified Linear Unit)

数据在大于0的时候，会保持原来的值，小于0直接变化为0

input 是需要

<img src="E:\notes\ML\images\pytorch\image-20220717011835984.png" alt="image-20220717011835984" style="zoom:50%;" />

参数： inplace 默认是False，经过ReLU之后input不会发生变化 ， 如果是True，会改变input为0. 一般用false， 防止数据丢失。

![image-20220717011531469](/D:/TyporaSpace/Images/image-20220717011531469-1683623934854-23.png)

把一个简单的tensor用ReLU处理

```python
import torch
from torch import  nn

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
print(input.shape)

class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.relu1 = nn.ReLU()

    def forward(self, x):
        output = self.relu1(x)
        return output

n = network()
output = n(input)
print(output)
```

引入数据集

```python
import torch
import torchvision.datasets
from torch import  nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./CIFAR", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, 64)



class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1= nn.Sigmoid()

    def forward(self, x):
        output = self.sigmoid1(x)
        return output

step = 0
n = network()
writer = SummaryWriter("./logs_relu")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = n(imgs)
    writer.add_images("output", output, global_step=step)
    step = step + 1

n = network()
output = n(input)
print(output)
```

sigmoid处理之后图像效果

<img src="E:\notes\ML\images\pytorch\image-20220717014543417.png" alt="image-20220717014543417" style="zoom: 67%;" />

#### 3.5 Normalization Layer 

正则化层：对数据进行正则化，网络出现overfit的时候会比较有效



#### 3.6 Recurrent Layer

主要包含了RNN, LSTM等能够在时序上对数据进行处理的模型



#### 3.7 Transformer Layer



#### 3.8 Linear Layer

线性层，也叫全连接层



<img src="E:\notes\ML\images\pytorch\LNN.webp" alt="LNN" style="zoom: 50%;" />



```python
torch.nn.Linear(in_features, # input layer 中神经元的个数
                out_features, # output layer 中神经元的个数
                bias=True, # w*x+b 如果为True则要加上这个bias
                device=None, dtype=None)
```

针对上图中的神经网络



#### 3.9 Dropout Layer

**作用：** 防止过拟合



#### 3.10  Sparse Layer

主要用于NLP中



#### IMPORTANT TIPS!!!!

https://blog.csdn.net/zxyhhjs2017/article/details/78605283



https://blog.csdn.net/qq_41318914/article/details/124135015



## 4. Loss, Backward, Optimizer, and Model

#### 4.1 Loss Function

**目的：** 计算目标和实际之间的差距，为更新提供依据（反向传播）

Tips: 使用的时候要更多的关注input，output的形状

##### L1Loss



##### MSELoss





##### CrossEntropyLoss



​	



## 5. Real Practice：复现



![See the source image](/D:/TyporaSpace/Images/Structure-of-CIFAR10-quick-model-1683623898314-20.png)

如果按照常规方式去写网络结构，需要init里面先声明一些variable(即网络layer)，

padding的计算要用到con里面具体的公式，其他的变量可以先假设为默认值

```python
from torch import nn
from torch.nn import Flatten, MaxPool2d, Conv2d, Linear
import torch


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = nn.Linear(1024, 64)
        self.linear2 = nn.Linear(64, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

n = Network()
print(n)

input = torch.ones([64, 3, 32, 32])
output = n(input)

print(output.shape)
```

`Sequential`的作用就是，把里面的**layer按照顺序去进行执行**， 节省代码量

```python
from torch import nn
from torch.nn import Flatten, MaxPool2d, Conv2d, Linear
import torch
from torch.utils.tensorboard import SummaryWriter


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model1 = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

n = Network()
print(n)

input = torch.ones([64, 3, 32, 32])
output = n(input)
print(output.shape)

writer = SummaryWriter("./logs_seq")
writer.add_graph(n, input)
writer.close()
```

`add_graph`是可以把**模型可视化**出来的



## 6. 使用现有的模型



## 7. 模型的保存与导入

### 方法一



### 方法二

