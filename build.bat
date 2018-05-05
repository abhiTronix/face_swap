mkdir build
cd build
set CMAKE_PREFIX_PATH=C:\face_swap\find_face_landmarks\build\install;C:\Users\%USERNAME%\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries;C:\face_swap\face_segmentation\build3\install;C:\face_Swap\dlib\build\install
cmake -C C:/Users/%USERNAME%/.caffe/dependencies/libraries_v140_x64_py27_1.1.0/libraries/caffe-builder-config.cmake .. -G "Visual Studio 14 2015 Win64" -DCMAKE_INSTALL_PREFIX=install -DCMAKE_BUILD_TYPE=Release -DWITH_BOOST_STATIC=OFF -DGLEW_LIBRARY=C:/face_swap/glew/lib/Release/x64 -DGLEW_INCLUDE_DIR=C:/face_swap/glew/include
