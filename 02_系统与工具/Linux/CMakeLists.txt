cmake_minimum_required(VERSION 3.12)
project(plan2_demo)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g")

set(OPENCV_INCLUDE_DIR /usr/include/opencv4)
set(OPENCV_LINK_DIR "")
set(OPENCV_LIBS opencv_imgcodecs opencv_imgproc opencv_core opencv_videoio opencv_highgui opencv_features2d opencv_calib3d)

set(PYTHON_INCLUDE_DIR /home/aidlux/.conda/envs/cpu/include/python3.8)
set(PYTHON_LIBRARY /home/aidlux/.conda/envs/cpu/lib/libpython3.8.so)

# 添加 libffi 库路径
set(FFI_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(FFI_LIBRARY /usr/lib/aarch64-linux-gnu/libffi.so)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${OPENCV_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIR}
    ${FFI_INCLUDE_DIR}  # 添加 libffi 头文件路径
)

aux_source_directory(. MAN_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC)

set(SOURCES ${MAN_SRC} ${SRC})

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_directories(${PROJECT_NAME} PUBLIC ${OPENCV_LINK_DIR})
target_link_libraries(${PROJECT_NAME}
    ${OPENCV_LIBS}
    ${PYTHON_LIBRARY}
    ${FFI_LIBRARY}  # 链接 libffi 库
)

message(STATUS "[CMAKEMSG] ${PROJECT_NAME} need libraries is : ${OPENCV_LIBS}")