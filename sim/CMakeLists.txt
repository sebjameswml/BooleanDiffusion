set(MORPH_LIBS_CORE ${ARMADILLO_LIBRARY} ${ARMADILLO_LIBRARIES} ${HDF5_C_LIBRARIES} ${LAPACK_LIBRARIES} jsoncpp_lib)
set(MORPH_LIBS_GL ${OpenCV_LIBS} OpenGL::GL Freetype::Freetype glfw)

add_executable(bd booldiffusion.cpp)
target_compile_definitions(bd PUBLIC FLT=float COMPILE_PLOTTING)
target_link_libraries(bd ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

# Compile compute-only with floats
add_executable(bdc booldiffusion.cpp)
target_compile_definitions(bdc PUBLIC FLT=float)
target_link_libraries(bdc ${MORPH_LIBS_CORE})
