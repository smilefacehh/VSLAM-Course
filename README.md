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
