CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(AGZ-SMAA)

############## options

OPTION(AGZ_SMAA_BUILD_EXAMPLE "build example application" OFF)

############## cmake

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
			"options: Debug Release RelWithDebInfo MinSizeRel" FORCE)
ENDIF()

############## library

SET(AGZ_SMAA_SRC
		"${PROJECT_SOURCE_DIR}/include/agz/smaa/mlaa.h"
		"${PROJECT_SOURCE_DIR}/include/agz/smaa/smaa.h"
		"${PROJECT_SOURCE_DIR}/src/smaa.cpp")

ADD_LIBRARY(AGZSMAA STATIC ${AGZ_SMAA_SRC})

TARGET_INCLUDE_DIRECTORIES(AGZSMAA PUBLIC "${PROJECT_SOURCE_DIR}/include")

############## example

IF(AGZ_SMAA_BUILD_EXAMPLE)
	ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/example")
ENDIF()
