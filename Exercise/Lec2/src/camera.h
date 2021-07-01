#pragma once

#include <string>
#include <memory>
#include <iostream>

namespace mango {

class Camera
{
public:
    Camera(int id, const std::string& name);
    ~Camera();

    /**
     * 读取yaml配置
    */
    void loadConfig(const std::string& file);

    /**
     * 打印相机名字
    */
    friend std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Camera>& cameraPtr)
    {
        os << cameraPtr->name() 
           << ", fx=" << cameraPtr->fx 
           << ", fy=" << cameraPtr->fy 
           << ", cx=" << cameraPtr->cx
           << ", cy=" << cameraPtr->cy 
           << ", k1=" << cameraPtr->k1
           << ", k2=" << cameraPtr->k2;
        return os;
    } 

    /**
     * 相机ID
    */
    inline int id()
    {
        return id_;
    }

    /**
     * 相机名字
    */
    inline std::string name()
    {
        return name_;
    }

    double fx, fy, cx, cy, k1, k2;   // 相机内参，畸变系数

private:
    int id_;                // 系统存在多个相机时，赋一个ID
    std::string name_;      // 相机名字
};

typedef std::shared_ptr<Camera> CameraPtr;

}