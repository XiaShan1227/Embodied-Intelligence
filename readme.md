## 1.RealSense

**1.1 RealSense文件架构**
```text
├── camera_extrinsics.npy
├── data
│   ├── depth
│   │   ├── depth_0.png
│   │   ├── depth_1.png
│   │   └── depth_2.png
│   ├── pointcloud
│   │   ├── pointcloud_0.ply
│   │   ├── pointcloud_1.ply
│   │   └── pointcloud_2.ply
│   └── rgb
│       ├── rgb_0.png
│       ├── rgb_1.png
│       └── rgb_2.png
├── extrinsics_from_quaternion.py
└── vision.py
```
使用RealSense 456相机采集数据，保存对应的RGB图、深度图和点云。
相机外参在采集数据的时候并没有使用，如果提供，可以打印查看相关信息。
运行 [extrinsics_from_quaternion.py](https://github.com/XiaShan1227/CDM-UASuction-RA/blob/master/RealSense/extrinsics_from_quaternion.py) 来从四元数生成`camera_extrinsics.npy`。

**1.2 执行`vision.py`来收集与可视化数据**
```bash
cd RealSense
python vision.py
```
**1.3 生成的数据点云如下：**
<p align="center">
  <img src="README/1.png" width="480" height="360"/>
</p>

**1.4 终端输出结果如下：**
<p align="center">
  <img src="README/2.png" width="480" height="360"/>
</p>

## 2.CDM

**2.1 CDM文件架构**
```text
├── depth_ref.py
├── infer.py
├── pretrain
│   └── cdm_d435.ckpt
├── README.md
├── rgbddepth
│   ├── dinov2_layers
│   │   ├── attention.py
│   │   ├── block.py
│   │   ├── drop_path.py
│   │   ├── __init__.py
│   │   ├── layer_scale.py
│   │   ├── mlp.py
│   │   ├── patch_embed.py
│   │   └── swiglu_ffn.py
│   ├── dinov2.py
│   ├── dpt.py
│   ├── __init__.py
│   └── util
│       ├── blocks.py
│       └── transform.py
└── setup.py
```

**2.2 执行`depth_ref.py`来修正相机深度图**
```bash
cd CDM
python depth_ref.py
```

**2.3 数据可视化如下：**
<p align="center">
  <img src="README/5.png" width="720" height="360" alt="数据可视化静态图"/>
</p>

<p align="center">
  <img src="README/4.gif" width="720" height="360" alt="数据可视化演示动画"/>
</p>

**2.4 终端输出结果如下：**
<p align="center">
  <img src="README/6.png" width="720" height="360" alt="终端输出结果"/>
</p>

## 3.Segmentation
```text
├── FastSAM
├── segmentation.py
├── server.py
└── ShenZhen
    ├── detect-trt-image.tar.gz
    ├── detect-trt-image-updated.tar.gz
    ├── readme.md
    ├── start.sh
    └── update.sh
```
Segmentation中提供了两个模型，一个弓叶科技深圳人工智能研究院提供的服务，一个为FastSAM模型。
`segmentation.py`中包括了如何使用这两个模型，包括分割单个物体或者多个物体，同时进行mask可视化。

**3.1 FastSAM生成实例分割mask**
```bash
cd Segmentation
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
pip install -e .
```
之后下载所需要的模型，并修改`segmentation.py`中的模型加载路径。

**3.2 使用服务端**  
<p> 首先需要启动docker服务:</p>

```bash
cd Segmentation/ShenZhen
sh start.sh
```

专门提供了一个`server.py`来调用服务并生成分割mask。
```bash
cd Segmentation
python server.py
```
分割mask示例：
<p align="center">
  <img src="RealSense/data/seg_mask_vis/vis_seg_mask_0.png" width="480" height="360"/>
</p>

<p>终端输出如下：</p>
<p align="center">
  <img src="README/7.png" width="720" height="360" alt="终端输出结果"/>
</p>

## 4.U_A_Suction
```text
├── MinkowskiEngine
├── model.py
├── pose_estimation.py
├── pose_single_object.py
├── pose_single_object_v1.py
├── pretrain
│   └── realsense.tar
└── utils.py
```
U_A_Suction使用深度学习模型来推理出物体吸取位姿。
`pose_estimation.py`可以推理并可视化多个物体的前Topk个位姿，`pose_single_object.py`仅仅可以推理并可视化1个物体的前Topk个位姿。

**4.1 安装MinkowskiEngine（cuda12.1）**  
```bash
cd U_A_Suction
git clone -b cuda-12-1 git@github.com:chenxi-wang/MinkowskiEngine.git
cd MinkowskiEngine
python setup.py install --verbose \
    --blas_include_dirs=${CONDA_PREFIX}/include \
    --blas_library_dirs=${CONDA_PREFIX}/lib \
    --blas=openblas
```
**4.2 执行`pose_estimation.py`**
```bash
cd U_A_Suction
python pose_estimation.py
```
<p align="center">
  <img src="README/8.png" width="480" height="360"/>
</p>
<p align="center">
  <img src="README/9.png" width="480" height="360"/>
</p>

**4.3 执行`pose_single_object.py`**
```bash
cd U_A_Suction
python pose_single_object.py
```
**4.4 实时推理** </br>
在`pose_single_object_v1.py`中提供了位姿估计实时推理的代码，可以直接接入RealSense相机进行数据读取。
同时在该脚本中增加了`_select_suction_center_and_direction()`函数。

本模块用于在点云中选择吸盘的中心点和吸取方向，综合考虑吸取分数和法向量与竖直方向的夹角。

---

(1) 吸盘中心点选择  

候选点集合为 $\{p_i\}_{i=1}^N$ (`top_k_points`)，对应吸取分数为 $\{s_i\}$ (`suction_scores`)。  

首先计算所有点对之间的欧式距离矩阵：  

$$
d_{ij} = \|p_i - p_j\|, \quad i,j = 1,\dots,N
$$  

定义邻域掩码：  

$$
M_{ij} =
\begin{cases}
1, & d_{ij} \le r \\
0, & \text{otherwise}
\end{cases}
$$  

其中 $r$ 为吸盘半径 (`suction_radius`)。  

每个候选点的邻域加权分数为：  

$$
w_i = \sum_{j=1}^{N} M_{ij} \cdot s_j
$$  

选择最大权重点作为中心点：  

$$
i^* = \mathrm{argmax}(w_i), \quad c = p_{i^*}
$$

其中 $c$ 对应 `center_point`。

---

(2) 吸取方向计算  

在中心点 $c$ 的邻域内，取对应法向量集合 $\{n_j\}$ (`top_k_normals[best_in_disk_idx]`) 和分数 $\{s_j\}$ (`suction_scores[best_in_disk_idx]`)。  

将法向量归一化：  

$$
\hat{n}_j = \frac{n_j}{\|n_j\|}
$$  

计算其与竖直方向 $z=(0,0,1)$ (`z_world`) 的夹角余弦：  

$$
\cos \theta_j = \max(0, \hat{n}_j \cdot z)
$$  

定义加权系数：  

$$
\alpha_j = s_j \cdot \Big(1 + \beta \cdot \cos \theta_j \Big)
$$  

其中 $\beta$ 为竖直方向偏置 (`vertical_bias`)。  

最终吸取方向为：  

$$
d = \frac{\sum_{j \in \text{disk}(i^{\ast})} \alpha_j \, n_j}{\left\|\sum_{j \in \text{disk}(i^{\ast})} \alpha_j \, n_j\right\|}
$$



其中 $d$ 对应 `suction_dir`。

---

最终返回结果：  

- 中心点 $c$ → `center_point`  
- 吸取方向 $d$ → `suction_dir`  
- 邻域内的分数集合 $\{s_j\}$ → `suction_scores[best_in_disk_idx]`
 




**4.5 使用CDM模型** <br>
可以修改`pose_estimation.py`或者`pose_single_object.py`中的深度图，将其输入到位姿估计模型。
<p align="center">
  <img src="README/10.png" width="720" height="360" alt="终端输出结果"/>
</p>

也可以直接使用`pose_single_object_v1.py`中已经修改好的代码
<p align="center">
  <img src="README/12.png" width="720" height="480" alt="终端输出结果"/>
</p>

<p align="center">
  <img src="README/11.png" width="480" height="360" alt="终端输出结果"/>
</p>


## 感谢
真挚感谢这些学者们对开源社区的贡献：
[CDM](https://github.com/ByteDance-Seed/manip-as-in-sim-suite) &nbsp;&nbsp;&nbsp;
[U_A_Suction](https://github.com/rcao-hk/UISN) &nbsp;&nbsp;&nbsp;
[FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) &nbsp;&nbsp;&nbsp;
[RealSense](https://github.com/IntelRealSense/librealsense) &nbsp;&nbsp;&nbsp;
[ROS2](https://github.com/ros2/ros2)