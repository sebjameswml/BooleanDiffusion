set(MORPH_LIBS_CORE ${ARMADILLO_LIBRARY} ${ARMADILLO_LIBRARIES} ${HDF5_C_LIBRARIES} ${LAPACK_LIBRARIES} jsoncpp_lib)
set(MORPH_LIBS_GL ${OpenCV_LIBS} OpenGL::GL Freetype::Freetype glfw)

add_executable(bd5 booldiffusion.cpp)
target_compile_definitions(bd5 PUBLIC FLT=float COMPILE_PLOTTING NGENES=5 NINPUTS=5)
target_link_libraries(bd5 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

# Compile compute-only with floats
add_executable(bd5c booldiffusion.cpp)
target_compile_definitions(bd5c PUBLIC FLT=float NGENES=5 NINPUTS=5)
target_link_libraries(bd5c ${MORPH_LIBS_CORE})

add_executable(bd2 booldiffusion.cpp)
target_compile_definitions(bd2 PUBLIC FLT=float COMPILE_PLOTTING NGENES=2 NINPUTS=2)
target_link_libraries(bd2 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

add_executable(bd2c booldiffusion.cpp)
target_compile_definitions(bd2c PUBLIC FLT=float NGENES=2 NINPUTS=2)
target_link_libraries(bd2c ${MORPH_LIBS_CORE})
