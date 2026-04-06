cmake_minimum_required(VERSION 3.14)
project(g1_bridge_sdk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

# unitree_sdk2
add_subdirectory(thirdparty/unitree_sdk2)

pybind11_add_module(g1_bridge_sdk MODULE src/g1_bridge.cpp)
target_link_libraries(g1_bridge_sdk PRIVATE pybind11::module unitree_sdk2)
target_compile_options(g1_bridge_sdk PRIVATE -O3)
