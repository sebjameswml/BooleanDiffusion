set(MORPH_LIBS_CORE ${ARMADILLO_LIBRARY} ${ARMADILLO_LIBRARIES} ${HDF5_C_LIBRARIES} ${LAPACK_LIBRARIES} jsoncpp_lib)
set(MORPH_LIBS_GL ${OpenCV_LIBS} OpenGL::GL Freetype::Freetype glfw)

add_executable(bd1_5 booldiffusion.cpp)
target_compile_definitions(bd1_5 PUBLIC FLT=float COMPILE_PLOTTING NGENES=5 NINPUTS=5)
target_link_libraries(bd1_5 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

# Compile compute-only with floats
#add_executable(bd1_5c booldiffusion.cpp)
#target_compile_definitions(bd1_5c PUBLIC FLT=float NGENES=5 NINPUTS=5)
#target_link_libraries(bd1_5c ${MORPH_LIBS_CORE})

add_executable(bd1_2 booldiffusion.cpp)
target_compile_definitions(bd1_2 PUBLIC FLT=float COMPILE_PLOTTING NGENES=2 NINPUTS=2)
target_link_libraries(bd1_2 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})

#add_executable(bd1_2c booldiffusion.cpp)
#target_compile_definitions(bd1_2c PUBLIC FLT=float NGENES=2 NINPUTS=2)
#target_link_libraries(bd1_2c ${MORPH_LIBS_CORE})

add_executable(bd2_2 booldiffusion.cpp)
target_compile_definitions(bd2_2 PUBLIC FLT=float COMPILE_PLOTTING NGENES=2 NINPUTS=2 BD_MARK2)
target_link_libraries(bd2_2 ${MORPH_LIBS_CORE} ${MORPH_LIBS_GL})
