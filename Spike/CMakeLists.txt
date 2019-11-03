file(GLOB_RECURSE SPIKE_MAIN_FILES
  ${PROJECT_SOURCE_DIR}/Spike/Backend/Context.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Helpers/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Models/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Neurons/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Plasticity/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/ActivityMonitor/*.cpp
  ${PROJECT_SOURCE_DIR}/Spike/Synapses/*.cpp
  )
file(GLOB_RECURSE SPIKE_CUDA_FILES ${PROJECT_SOURCE_DIR}/Spike/Backend/*.cu)
file(GLOB_RECURSE SPIKE_DUMMY_FILES ${PROJECT_SOURCE_DIR}/Spike/Backend/Dummy/*.cpp)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DSPIKE_DEFAULT_BACKEND="CUDA")
cuda_add_library(Spike ${SPIKE_MAIN_FILES} ${SPIKE_CUDA_FILES} ${SPIKE_DUMMY_FILES})
set_property(TARGET Spike PROPERTY CUDA_STANDARD 11)


