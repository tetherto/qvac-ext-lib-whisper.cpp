find_package(Threads REQUIRED)

set(TEST_TARGET test-backend-ops)
add_executable(${TEST_TARGET} ${TEST_TARGET}.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml Threads::Threads)
add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
