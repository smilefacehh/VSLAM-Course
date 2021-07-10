#pragma once

#include "camera.h"
#include <map>

namespace mango {
class CameraMgr
{
public:
    static CameraMgr& getInstance()
    {
        static CameraMgr cameraMgr;
        return cameraMgr;
    }

    ~CameraMgr();

    /**
     * 添加一个相机
    */
    void addCamera(CameraPtr& camera);

    /**
     * 根据ID获取相机
    */
    CameraPtr& getCameraById(int id);

private:
    CameraMgr();
    CameraMgr(const CameraMgr& rhs);
    CameraMgr& operator=(const CameraMgr& rhs);

    std::map<int, CameraPtr> camera_ids_;
};

}