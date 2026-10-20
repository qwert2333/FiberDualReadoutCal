set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(Simulation)

find_package(Geant4 REQUIRED ui_all vis_all)

# Find ROOT
find_package(ROOT REQUIRED COMPONENTS RIO Tree Hist Core)
include(${ROOT_USE_FILE})


include(${Geant4_USE_FILE})
include_directories(${PROJECT_SOURCE_DIR}/include) # needed to let compiler know where header files are.  
file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers ${PROJECT_SOURCE_DIR}/include/*.hh)
file(GLOB macros  ${PROJECT_SOURCE_DIR}/macro)
file(GLOB run_macros ${PROJECT_SOURCE_DIR}/run)
file(GLOB jupyter_analysises ${PROJECT_SOURCE_DIR}/jupyter_analysis)
file(GLOB meshes ${PROJECT_SOURCE_DIR}/stl_files)

file(COPY ${macros} DESTINATION ${PROJECT_BINARY_DIR})
file(COPY ${run_macros} DESTINATION ${PROJECT_BINARY_DIR})
file(COPY ${jupyter_analysises} DESTINATION ${PROJECT_BINARY_DIR})
file(COPY ${meshes} DESTINATION ${PROJECT_BINARY_DIR})

add_executable(main main.cc ${sources} ${headers})
target_link_libraries(main ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})
add_custom_target(Simulation DEPENDS main)
