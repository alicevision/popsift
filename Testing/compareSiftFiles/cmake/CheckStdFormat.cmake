# Chooses the formatting library of compareSiftFiles.
#
# std::format is used if <format> compiles in C++20 and defines
# __cpp_lib_format, which also rules out standard libraries that ship an
# incomplete <format> (libstdc++ before 13, libc++ before 17, or macOS
# deployment targets without floating-point std::to_chars).
# Otherwise compareSiftFiles stays at C++17 and uses fmtlib: an installed
# fmt package if there is one, else fmt fetched with FetchContent.
#
# Sets
#   Csf_HAVE_STD_FORMAT  1 for std::format, 0 for fmtlib
#   Csf_CXX_STANDARD     20 with std::format, 17 with fmtlib
# and creates the target fmt::fmt if fmtlib is used.
#
# To test the fmtlib path with a compiler that has std::format, preset the
# cached result of the check: -DCsf_STD_FORMAT_COMPILES=OFF

include(CheckCXXSourceCompiles)

function(csf_check_std_format)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  check_cxx_source_compiles("
    #include <format>
    #include <string>
    #ifndef __cpp_lib_format
    #error \"<format> without __cpp_lib_format\"
    #endif
    int main()
    {
      const std::string s = std::format( \"{},{:.6g},{}\", 42, 1.5f, std::string_view(\"x\") );
      return s.empty() ? 1 : 0;
    }"
    Csf_STD_FORMAT_COMPILES)
  if(Csf_STD_FORMAT_COMPILES)
    set(Csf_HAVE_STD_FORMAT 1 PARENT_SCOPE)
  else()
    set(Csf_HAVE_STD_FORMAT 0 PARENT_SCOPE)
  endif()
endfunction()

csf_check_std_format()

if(Csf_HAVE_STD_FORMAT)
  set(Csf_CXX_STANDARD 20)
  message(STATUS "compareSiftFiles uses std::format")
else()
  set(Csf_CXX_STANDARD 17)
  # fmt 8 is the first version that checks format strings at compile time,
  # like std::format.
  find_package(fmt 8 QUIET CONFIG)
  if(fmt_FOUND)
    message(STATUS "compareSiftFiles uses fmtlib ${fmt_VERSION} from ${fmt_DIR}")
  else()
    message(STATUS "compareSiftFiles uses fmtlib, fetching fmt 11.2.0")
    include(FetchContent)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    FetchContent_Declare(fmt
      URL      https://github.com/fmtlib/fmt/releases/download/11.2.0/fmt-11.2.0.zip
      URL_HASH SHA256=203eb4e8aa0d746c62d8f903df58e0419e3751591bb53ff971096eaa0ebd4ec3)
    FetchContent_MakeAvailable(fmt)
  endif()
endif()
