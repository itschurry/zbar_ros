cmake_minimum_required(VERSION 3.5)
project(zbar_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(zbar_ros_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# Include header files
include_directories(
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
)

# Build library
add_library(barcode_reader_node src/barcode_reader_node.cpp)
ament_target_dependencies(barcode_reader_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  zbar_ros_interfaces)
target_link_libraries(barcode_reader_node zbar)
target_link_libraries(barcode_reader_node ${OpenCV_LIBRARIES})

# Build executable
add_executable(barcode_reader src/barcode_reader_main.cpp)
target_link_libraries(barcode_reader barcode_reader_node)

# Install
install(TARGETS
  barcode_reader
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/
  DESTINATION include/)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})
install(
  TARGETS barcode_reader_node
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

# Test
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Skip copyright check, because the copyright used is not recognised by ament_copyright
  set(ament_cmake_copyright_FOUND TRUE)

  ament_lint_auto_find_test_dependencies()
endif()

# Package
ament_package()
