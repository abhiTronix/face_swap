# Target
add_executable(apply_w apply_w.cpp)

target_include_directories(apply_w PUBLIC
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(apply_w PUBLIC
	${OpenCV_LIBS}
)

# Installations
install(TARGETS apply_w DESTINATION bin COMPONENT app)

