# VSLAM-Course
http://rpg.ifi.uzh.ch/teaching.html

每章节代码递增包含前一章节的代码，均为相对底层的实现，非接口调用。

## Lec2 相机模型
功能
- 图像畸变矫正
- 已知相机内参、外参，在棋盘格上绘制立方体

运行
```
cd Lec2
mkdir build
cd build
cmake ..
make -j8
./draw_cube ../data/ ../conf/
```

![image](https://github.com/smilefacehh/VSLAM-Course/blob/main/Lec2/cube.png)

## Lec3 相机标定
功能
- 已知相机内参，世界点-像素点匹配对，DLT求解位姿R、t

运行
```
cd Lec3
mkdir build
cd build
cmake ..
make -j8
./draw_cube ../data/ ../conf/
```

![image](https://github.com/smilefacehh/VSLAM-Course/blob/main/Lec3/calib.png)

## Lec5 harris角点提取与跟踪
功能
- 提取harris角点（详细流程代码）
- 计算patch描述子
- 描述子匹配跟踪

运行
```
cd Lec5
mkdir build
cd build
cmake ..
make -j8
./harris_track ../data/
```

![image](https://github.com/smilefacehh/VSLAM-Course/blob/main/Lec5/harri-track.png)

## Lec6 sift特征点
```diff
- 注：结果不对，还未调试完全
```
功能
- 提取sift特征点（DoG）
- HoG梯度直方图描述子
- 匹配

运行
```
cd Lec6
mkdir build
cd build
cmake ..
make -j8
./sift_match ../data/
```

![image](https://github.com/smilefacehh/VSLAM-Course/blob/main/Lec6/output/match.png)

## Lec7 双目视差
```diff
- 注：中间结果是对的，最后拼接的点云似乎效果不对
```
功能
- 对齐的双目图像计算视差
- 视差计算3D坐标，拼接点云

运行
```
cd Lec7
mkdir build
cd build
cmake ..
make -j8
./stereo_vision ../data/
```
所有点计算视差，不做额外处理的视差图
![image](https://github.com/smilefacehh/VSLAM-Course/blob/main/Lec7/output/unfiltered_disp.png)

处理之后的视差图，对于某个点如果超过3个ssd比较小的匹配结果，那么认为匹配不够准确，视差置0
![image](https://github.com/smilefacehh/VSLAM-Course/blob/main/Lec7/output/filtered_disp.png)

视差图对应的深度图
![image](https://github.com/smilefacehh/VSLAM-Course/blob/main/Lec7/output/depth.png)

## Lec8 双目外参估计
```diff
- 注：还没完全调对…
```
功能
- 已知匹配像素点，八点法计算基础矩阵F
- 基础矩阵分解得到R、t，通过三角化选择正确的解
- 归一化八点法计算基础矩阵F
- 三角化
- 基础矩阵与本质矩阵的转换
- 误差度量

运行
```
cd Lec8
mkdir build
cd build
cmake ..
make -j8
./stereo_vision ../data/
```

## 踩坑

- cv::Mat.at<T> T的类型一定要对，否则会出问题
- Eigen::Matrix3f F; 要指明了矩阵的维度，才能使用 <<
- Eigen::MatrixXf A; A.resize(a,b); 没有指明维度的矩阵，需要resize，才能赋值
 

## 代码规范

- camera_mgr.h，文件名，小写、下划线
- class CameraMgr，类名，大写、驼峰
- loadConfig，类成员函数，小写、驼峰
- load_matrix，普通函数，小写、下划线
- camera_ids_，类私有成员变量，小写、下划线、结尾下划线
- fx，类公有成员变量，小写
- blockSize，函数参数，小写、驼峰
- tmp_val，局部变量，小写、驼峰
- 缩进，4空格
- 注释，类、方法，用/***/，标注参数、返回值；成员变量用//行尾注释，过长则单独一行
- (a + b)*(a - c)，乘法不空格，加减空格
- 0.2f，小数默认是double类型，能带f尽量带上
