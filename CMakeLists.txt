CMAKE_MINIMUM_REQUIRED(VERSION 3.10)

PROJECT(AGZ-SMAA)

############## options

OPTION(AGZ_SMAA_BUILD_EXAMPLE "build example application" 				     OFF)
OPTION(AGZ_SMAA_STATIC        "build static library (instead of shared lib)" ON)

############## cmake

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "options: Debug Release RelWithDebInfo MinSizeRel" FORCE)
ENDIF()

############## library

SET(AGZ_SMAA_SRC
	"${PROJECT_SOURCE_DIR}/include/agz/smaa/mlaa.h"
	"${PROJECT_SOURCE_DIR}/src/mlaa.cpp"
	"${PROJECT_SOURCE_DIR}/src/helper.h"
	"${PROJECT_SOURCE_DIR}/src/mlaa_blend.inl"
	"${PROJECT_SOURCE_DIR}/src/mlaa_depth_edge.inl"
	"${PROJECT_SOURCE_DIR}/src/mlaa_lum_edge.inl"
	"${PROJECT_SOURCE_DIR}/src/mlaa_vs.inl"
	"${PROJECT_SOURCE_DIR}/src/mlaa_weight.inl")

IF(AGZ_SMAA_STATIC)
	SET(AGZ_SMAA_LIB_TYPE STATIC)
ELSE()
	SET(AGZ_SMAA_LIB_TYPE SHARED)
ENDIF()

ADD_LIBRARY(AGZSMAA ${AGZ_SMAA_LIB_TYPE} ${AGZ_SMAA_SRC})

TARGET_INCLUDE_DIRECTORIES(AGZSMAA PUBLIC "${PROJECT_SOURCE_DIR}/include")

############## example

IF(AGZ_SMAA_BUILD_EXAMPLE)
	ADD_SUBDIRECTORY("${PROJECT_SOURCE_DIR}/example")
ENDIF()
