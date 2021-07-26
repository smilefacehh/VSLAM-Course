#include "camera_mgr.h"

namespace mango {

CameraMgr::CameraMgr(){}
CameraMgr::CameraMgr(const CameraMgr& rhs) {}
CameraMgr& CameraMgr::operator=(const CameraMgr& rhs)
{
    return *this;
}

CameraMgr::~CameraMgr(){}

void CameraMgr::addCamera(CameraPtr& camera)
{
    if(camera_ids_.find(camera->id()) != camera_ids_.end())
    {
        camera_ids_[camera->id()] = camera;
    }
    else
    {
        camera_ids_.insert(std::pair<int, CameraPtr>(camera->id(), camera));
    }
}
    
CameraPtr& CameraMgr::getCameraById(int id)
{
    if(camera_ids_.find(id) != camera_ids_.end())
    {
        return camera_ids_[id];
    }
    CameraPtr camera = std::shared_ptr<Camera>(new Camera(id, "new one"));
    addCamera(camera);
    return camera_ids_[id];
}

}