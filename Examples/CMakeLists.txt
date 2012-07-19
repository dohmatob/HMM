add_executable(martian martian.cpp)
target_link_libraries(martian HMM)

add_executable(dummy dummy.cpp)
target_link_libraries(dummy HMM)
