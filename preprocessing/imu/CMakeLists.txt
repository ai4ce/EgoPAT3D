# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
find_package(k4a)
find_package(k4arecord)

add_executable(imu main.c)
target_link_libraries(imu 
    k4a::k4a
    k4a::k4arecord
    )
