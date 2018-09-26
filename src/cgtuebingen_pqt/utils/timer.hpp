#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <stdexcept>


template<typename C = std::chrono::high_resolution_clock>
class timer
{
   std::chrono::time_point<C> start_v;
   std::chrono::time_point<C> pause_v;
   std::chrono::time_point<C> end_v;

  bool stopped;
  bool paused;

public:
  explicit timer() : stopped(false), paused(false){
    start_v = C::now();
  }
  ~timer(){}

  /**
   * @brief start stopwatch
   * @details [long description]
   */
  void start(){
    start_v = C::now();
    paused = false;
    stopped = false;
  }

  /**
   * @brief break stopwatch, but allows resuming
   * @details [long description]
   */
  void pause(){
    pause_v = C::now();
    paused = true;
  }

  void resume(){
    if(stopped)
      throw std::runtime_error("cannot resume a stopped timer");
    start_v += C::now() - pause_v;
    paused = false;
    stopped = false;
  }

  void stop(){
    end_v = C::now();
    stopped = true;
  }

  template<typename U = std::chrono::milliseconds>
  typename U::rep elapsed() const
  {
    /*

    example:
      std::cout << t.elapsed<std::chrono::nanoseconds>() << std::endl;

    std::chrono::nanoseconds
    std::chrono::microseconds
    std::chrono::milliseconds
    std::chrono::seconds
    std::chrono::minutes
    std::chrono::hours
    */

      return
        (stopped) ?
          std::chrono::duration_cast<U>(end_v - start_v).count() :
          (paused) ?
            std::chrono::duration_cast<U>(pause_v - start_v).count() :
            std::chrono::duration_cast<U>(C::now() - start_v).count();;
  }

};

#endif

