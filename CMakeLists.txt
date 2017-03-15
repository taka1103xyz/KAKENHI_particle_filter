cmake_minimum_required(VERSION 2.8)
project( Particle )
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
endif()

include_directories(/usr/include/eigen3)

add_executable( Particle particle.cpp )
