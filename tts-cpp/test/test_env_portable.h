#pragma once
// Portable setenv/unsetenv for the test harnesses. MSVC's CRT has no
// POSIX setenv/unsetenv (only _putenv_s), so provide same-named shims
// there; every other platform gets the real functions from <cstdlib>.
#include <cstdlib>
#ifdef _MSC_VER
inline int setenv(const char * name, const char * value, int /*overwrite*/) {
    return _putenv_s(name, value);
}
inline int unsetenv(const char * name) {
    return _putenv_s(name, ""); // empty value removes the variable
}
#endif
