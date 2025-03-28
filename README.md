# ROS小车巡线程序

https://github.com/user-attachments/assets/28b3e789-3a88-4af8-8846-142becb5ed8b

## 环境安装
-安装OpenCV及其他依赖库：可以通过pip工具安装，例如：
```bash
  pip install opencv-python opencv-contrib-python ros-melodic-joy ros-melodic-teleop-twist-joy ros-melodic-teleop-twist-keyboard ros-melodic-laser-proc ros-melodic-rgbd-launch ros-melodic-depthimage-to-laserscan ros-melodic-amcl ros-melodic-map-server ros-melodic-move-base ros-melodic-urdf ros-melodic-xacro ros-melodic-compressed-image-transport ros-melodic-rqt-image-view ros-melodic-gmapping ros-melodic-navigation ros-melodic-rosserial-arduino ros-melodic-rosserial-python ros-melodic-rosserial-server ros-melodic-rosserial-client ros-melodic-rosserial-msgs ros-melodic-turtlebot3*
```
## 运行方法
-创建robots工作空间
```bash
~$ mkdir -p ~/robots/src
~$ cd robots/src
~/robots/src$ catkin_init_workspace
~/robots/src$ cd ..
~/robots$ catkin_make
~/robots$ echo "source ~/robots/devel/setup.bash" >> ~/.bashrc
```
-环境部分完成
```bash
# 打开一个终端
export TURTLEBOT3_MODEL=waffle	#加载机器人
roslaunch followbot turtlebot3_course.launch #加载地图
# 打开第二个终端
rosrun followbot follower.py
```
# 参考
<p>https://zhuanlan.zhihu.com/p/8549697809</p>
<p>https://gitee.com/webcohort/ros</p>
<p>https://zhuanlan.zhihu.com/p/664776413</p>
