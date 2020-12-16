set(MORPH_LIBS_CORE ${ARMADILLO_LIBRARY} ${ARMADILLO_LIBRARIES} ${HDF5_C_LIBRARIES} ${LAPACK_LIBRARIES} jsoncpp_lib)
set(MORPH_LIBS_GL ${OpenCV_LIBS} OpenGL::GL Freetype::Freetype glfw)

set(COMPILE_FIRST 0)
set(COMPILE_SECOND 0)
set(COMPILE_THIRD 1)

# First system
if(COMPILE_FIRST)
  add_executable(bd1_5 booldiffusion.cpp)
  target_compile_definitions(bd1_5 PUBLIC FLT=float COMPILE_PLOTTING NGENES=5 NINPUTS=5)
  target_link_libraries(bd1_5 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

  add_executable(bd1_2 booldiffusion.cpp)
  target_compile_definitions(bd1_2 PUBLIC FLT=float COMPILE_PLOTTING NGENES=2 NINPUTS=2)
  target_link_libraries(bd1_2 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})
endif()

# Second system
if(COMPILE_SECOND)
  add_executable(bd2_2 booldiffusion.cpp)
  target_compile_definitions(bd2_2 PUBLIC FLT=float COMPILE_PLOTTING NGENES=2 NINPUTS=2 BD_MARK2)
  target_link_libraries(bd2_2 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

  add_executable(bd2_2c booldiffusion.cpp)
  target_compile_definitions(bd2_2c PUBLIC FLT=float NGENES=2 NINPUTS=2 BD_MARK2)
  target_link_libraries(bd2_2c ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

  add_executable(bd2_3 booldiffusion.cpp)
  target_compile_definitions(bd2_3 PUBLIC FLT=float COMPILE_PLOTTING NGENES=3 NINPUTS=3 BD_MARK2)
  target_link_libraries(bd2_3 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

  add_executable(bd2_5 booldiffusion.cpp)
  target_compile_definitions(bd2_5 PUBLIC FLT=float COMPILE_PLOTTING NGENES=5 NINPUTS=5 BD_MARK2)
  target_link_libraries(bd2_5 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})
endif()

# Third system
add_executable(bd3_2 booldiffusion.cpp)
target_compile_definitions(bd3_2 PUBLIC FLT=float COMPILE_PLOTTING NGENES=2 NINPUTS=2 BD_MARK3)
target_link_libraries(bd3_2 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

add_executable(bd3_2c booldiffusion.cpp)
target_compile_definitions(bd3_2c PUBLIC FLT=float NGENES=2 NINPUTS=2 BD_MARK3)
target_link_libraries(bd3_2c ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

add_executable(bd3_3 booldiffusion.cpp)
target_compile_definitions(bd3_3 PUBLIC FLT=float COMPILE_PLOTTING NGENES=3 NINPUTS=3 BD_MARK3)
target_link_libraries(bd3_3 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})
