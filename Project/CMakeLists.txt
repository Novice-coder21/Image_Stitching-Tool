cmake_minimum_required(VERSION 3.5.0)
project(project VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(project main.cpp MyBlending.cpp MyMatching.cpp MySift.cpp)

# Link against OpenCV and libjpeg-turbo libraries
target_link_libraries(project ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)