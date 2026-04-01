cmake_minimum_required(VERSION 3.10)

find_package(Git QUIET)

if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --long --match "v*"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DESCRIBE_OUTPUT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(GIT_DESCRIBE_OUTPUT MATCHES "^v([0-9]+\\.[0-9]+\\.[0-9]+)-([0-9]+)-g([0-9a-f]+)$")
        set(GIT_VERSION "${CMAKE_MATCH_1}")
        set(GIT_DISTANCE "${CMAKE_MATCH_2}")
        set(GIT_HASH "${CMAKE_MATCH_3}")
        if(GIT_DISTANCE STREQUAL "0")
            set(DETECTED_VERSION "${GIT_VERSION}")
        else()
            set(DETECTED_VERSION "${GIT_VERSION}+git${GIT_HASH}")
        endif()
        message(STATUS "Version from git tag: ${DETECTED_VERSION}")
    endif()
endif()

if(NOT DEFINED DETECTED_VERSION OR DETECTED_VERSION STREQUAL "")
    set(DETECTED_VERSION "0.0.0")
    message(WARNING "Could not detect version from git tag, using fallback: ${DETECTED_VERSION}")
endif()

# Strip +gitXXXXXXX suffix for CMake project VERSION (must be numeric X.Y.Z)
if(DETECTED_VERSION MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)")
    set(NUMERIC_VERSION "${CMAKE_MATCH_1}")
else()
    set(NUMERIC_VERSION "0.0.0")
endif()

project(MainProject VERSION ${NUMERIC_VERSION})
message(STATUS "Project version: ${PROJECT_VERSION} (full: ${DETECTED_VERSION})")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_BINARY_DIR}/include)

file(MAKE_DIRECTORY ${CMAKE_INSTALL_INCLUDEDIR})

file(GLOB XXHASH_HEADERS "third_party/xxHash/*.h")
file(COPY ${XXHASH_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(XXHASH_SOURCES
    third_party/xxHash/xxhash.c
)
add_library(xxhash SHARED ${XXHASH_SOURCES})
target_include_directories(xxhash PUBLIC
    third_party/xxHash
    ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${XXHASH_HEADERS} DESTINATION include)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
