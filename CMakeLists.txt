cmake_minimum_required(VERSION 3.15)
project(MyProject LANGUAGES CXX CUDA)

# 设置 CUDA 编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 启用 CUDA 支持
enable_language(CUDA)

# 指定 CUDA 架构（A100 使用 compute_80）
set(CMAKE_CUDA_ARCHITECTURES 80)

# 自动查找所有子目录中的 .cu 文件
file(GLOB_RECURSE ALL_CU_FILES
    "example/**/*.cu"
)

# 遍历每个 .cu 文件并创建一个可执行文件
foreach(CU_FILE ${ALL_CU_FILES})
    # 获取文件名，不包括扩展名
    get_filename_component(EXEC_NAME ${CU_FILE} NAME_WE)

    # 创建可执行文件
    add_executable(${EXEC_NAME} ${CU_FILE})

    # 设置 CUDA 编译选项
    target_compile_features(${EXEC_NAME} PRIVATE cxx_std_14)
endforeach()
