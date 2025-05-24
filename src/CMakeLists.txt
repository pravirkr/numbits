# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(numbits_python numbits_bindings.cpp)
set_target_properties(numbits_python PROPERTIES OUTPUT_NAME "numbits")

# Link against the C++ header-only library
target_link_libraries(numbits_python PRIVATE numbits::numbits)
target_include_directories(numbits_python PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Install directive for scikit-build-core
install(TARGETS numbits_python LIBRARY DESTINATION .)
