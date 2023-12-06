cmake_minimum_required(VERSION 3.1.2)
project(d2p LANGUAGES C CXX)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  depth_image_proc
  eigen_conversions
  geometry_msgs
  image_geometry
  roscpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
)
catkin_package()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR} 
)

add_executable(${PROJECT_NAME}_node
  src/d2p_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)