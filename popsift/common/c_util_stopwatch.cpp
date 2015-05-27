#include "c_util_stopwatch.hpp"

StopWatch::StopWatch()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;
#ifdef _WIN32
    freq_set = false;
    if (!freq_set) {
        LARGE_INTEGER temp;
        QueryPerformanceFrequency((LARGE_INTEGER *) & temp);
        freq = ((double) temp.QuadPart) / 1000.0;
        freq_set = true;
    }
#endif
}

StopWatch::~StopWatch()
{
}

void StopWatch::start()
{
#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER *) & start_time);
#else
    gettimeofday(&start_time, 0);
#endif
    running = true;
}

void StopWatch::stop()
{
#ifdef _WIN32
    QueryPerformanceCounter((LARGE_INTEGER *) & end_time);
    diff_time =
        (float) (((double) end_time.QuadPart -
                  (double) start_time.QuadPart) / freq);
#else
    gettimeofday(&t_time, 0);
    diff_time = (float) (1000.0 * (t_time.tv_sec - start_time.tv_sec)
                         +
                         (0.001 * (t_time.tv_usec - start_time.tv_usec)));
#endif
    total_time += diff_time;
    running = false;
    clock_sessions++;
}

void StopWatch::reset()
{
    diff_time = 0;
    total_time = 0;
    clock_sessions = 0;

    if (running) {
#ifdef _WIN32
        QueryPerformanceCounter((LARGE_INTEGER *) & start_time);
#else
        gettimeofday(&start_time, 0);
#endif
    }
}

int StopWatch::print_total_time(const char *str)
{
    fprintf(stderr, "Time ( %s ) = %.2lf msec\n", str, get_time());
    return 1;
}

int StopWatch::print_diff_time(const char *str)
{
    fprintf(stderr, "Diff Time ( %s ) = %.2lf msec\n", str, get_diff_time());
    return 1;
}

int StopWatch::print_average_time(const char *str)
{
    fprintf(stderr, "Avg Time of %d runs ( %s ) = %.2lf msec\n", clock_sessions, str,
            get_average_time());
    return 1;
}

float StopWatch::get_time() const
{
    // Return the TOTAL time to date
    float retval = total_time;

    if (running) {
        retval += get_diff_time();
    }

    return retval;
}

float StopWatch::get_diff_time() const
{
#ifdef _WIN32
    LARGE_INTEGER temp;
    QueryPerformanceCounter((LARGE_INTEGER *) & temp);
    return (float) (((double) (temp.QuadPart - start_time.QuadPart)) /
                    freq);
#else
    struct timeval t_time;
    gettimeofday(&t_time, 0);
    return (float) (1000.0 * (t_time.tv_sec - start_time.tv_sec)
                    + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
#endif
}

float StopWatch::get_average_time() const
{
    return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}

