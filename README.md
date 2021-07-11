# VSLAM-Course
## 相机模型
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

## 相机标定
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
