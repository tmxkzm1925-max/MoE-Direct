set(TARGET moe-direct-selftest)
add_executable(${TARGET} moe-direct-selftest.cpp selftest-harness.cpp suite-memplan-s2.cpp selftest-lm-s0.cpp)
# [MoE-Direct 레버 A 단계 1 / S1] memplan selfcheck 진입점(llama_memplan_selfcheck)이 llama 안에 있다.
target_link_libraries(${TARGET} PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ../../vendor ../../src)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
