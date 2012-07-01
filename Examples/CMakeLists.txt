add_executable(martian martian.cpp)
target_link_libraries(martian HMM)

add_executable(letters letters.cpp)
target_link_libraries(letters HMM)