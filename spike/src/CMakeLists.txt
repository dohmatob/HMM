file(GLOB sources *.cpp)
include_directories(${HMM_SOURCE_DIR})

# Install headers
INSTALL(FILES ${sources} DESTINATION include/HMM)