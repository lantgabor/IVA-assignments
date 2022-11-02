cmake_minimum_required(VERSION 3.9)
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:WINDOWS")

project( main )


set(OpenCV_STATIC ON)
find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )

# include_directories(include)
add_executable( main src/main.cpp )
target_link_libraries( main ${OpenCV_LIBS}) 

# linker flag for no console: /ENTRY:"mainCRTStartup"

set(CMAKE_INSTALL_PREFIX install )
install(TARGETS main DESTINATION ${CMAKE_INSTALL_PREFIX})