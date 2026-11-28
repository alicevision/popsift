# Sets Csf_HAVE_PAR_EXECUTION to 1 if the C++17 parallel algorithms
# (std::execution::par_unseq) compile and link, otherwise to 0.
# libstdc++ runs them on TBB, so TBB is searched and linked into the test
# when it is installed. Without TBB, libstdc++ 9 cannot compile <execution>,
# while newer versions run the algorithms serially.

include(CheckCXXSourceCompiles)

find_package(TBB QUIET CONFIG)

function(csf_check_par_execution)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(TBB_FOUND)
    set(CMAKE_REQUIRED_LIBRARIES TBB::tbb)
  endif()
  check_cxx_source_compiles("
    #include <algorithm>
    #include <execution>
    #include <vector>
    int main()
    {
      std::vector<int> v( 100, 1 );
      std::for_each( std::execution::par_unseq, v.begin(), v.end(),
                     []( int& i ) { i *= 2; } );
      return v[0] == 2 ? 0 : 1;
    }"
    Csf_PAR_EXECUTION_COMPILES)
  if(Csf_PAR_EXECUTION_COMPILES)
    set(Csf_HAVE_PAR_EXECUTION 1 PARENT_SCOPE)
  else()
    set(Csf_HAVE_PAR_EXECUTION 0 PARENT_SCOPE)
  endif()
endfunction()

csf_check_par_execution()
