#include "string_util.h"

namespace mango {
/**
 * 目录路径末尾添加斜杠/
 * @param folder 目录路径
*/
std::string folder_add_slash(const std::string& folder)
{
    size_t len = folder.length();
    if(len > 0 && folder[len - 1] != '/')
    {
        return folder + '/';
    }
    return folder;
}

}