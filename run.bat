set PATH=c:\face_swap\glew\bin\Release\x64;C:\Users\%USERNAME%\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\bin;C:\Program Files (x86)\common files\intel\Shared Libraries\redist\intel64_win\compiler;C:\Users\%USERNAME%\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\x64\vc14\bin;C:\Users\%USERNAME%\.caffe\dependencies\libraries_v140_x64_py27_1.1.0\libraries\lib;C:\face_swap\find_face_landmarks\build\install\bin;%PATH%
set Qt5Widgets_DIR=C:\QT_\5.6.0\msvc2015_64
set QT_QPA_PLATFORM_PLUGIN_PATH=C:\QT_\5.6.0\msvc2015_64\plugins\platforms

cd /d C:\face_swap\face_swap\build\install\bin
face_swap_image --cfg test.cfg
