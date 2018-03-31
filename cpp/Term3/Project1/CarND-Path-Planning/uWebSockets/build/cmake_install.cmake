# Install script for directory: /Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/lib/libuWS.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/lib" TYPE SHARED_LIBRARY FILES "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/build/libuWS.dylib")
  if(EXISTS "$ENV{DESTDIR}/usr/local/lib/libuWS.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/usr/local/lib/libuWS.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/strip" "$ENV{DESTDIR}/usr/local/lib/libuWS.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/usr/local/include/uWS/Extensions.h;/usr/local/include/uWS/WebSocketProtocol.h;/usr/local/include/uWS/Networking.h;/usr/local/include/uWS/WebSocket.h;/usr/local/include/uWS/Hub.h;/usr/local/include/uWS/Group.h;/usr/local/include/uWS/Node.h;/usr/local/include/uWS/Socket.h;/usr/local/include/uWS/HTTPSocket.h;/usr/local/include/uWS/uWS.h;/usr/local/include/uWS/uUV.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/usr/local/include/uWS" TYPE FILE FILES
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/Extensions.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/WebSocketProtocol.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/Networking.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/WebSocket.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/Hub.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/Group.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/Node.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/Socket.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/HTTPSocket.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/uWS.h"
    "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/src/uUV.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/build/examples/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/Users/saajan/Documents/CarND/cpp/Term3/Project1/CarND-Path-Planning/uWebSockets/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
