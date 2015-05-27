#pragma once

#include <iostream>
#include <cstdio>

#include "c_util.hpp"

// includes, system
#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#else
#include <ctime>
#include <sys/time.h>
#endif

/*********************************************************/
/* @class name StopWatch                                 */
/* @brief Time measurement utility for windows and linux */
/*********************************************************/
class StopWatch {

public:

    /* Constructor and destructor */
    StopWatch();
    ~StopWatch();

    /* Start time measurement */
    void start();

    /* Stop time measurement */
    void stop();

    /* Reset time counters to zero */
    void reset();

    /* Print the total execution time  */
    int print_total_time(const char *str);

    /* Print the diff execution time  */
    int print_diff_time(const char *str);

    /* Print the average execution time  */
    int print_average_time(const char *str);

    /* Time in msec. after start. If the stop watch is still running (i.e. there */
    /* was no call to stop()) then the elapsed time is returned, otherwise the */
    /* time between the last start() and stop call is returned */
    float get_time() const;

    float get_diff_time() const;

    /* Mean time to date based on the number of times the stopwatch has been */
    /* _stopped_ (ie finished sessions) and the current total time */
    float get_average_time() const;

    inline StopWatch & operator=(const float tm);
    inline StopWatch & operator+=(const float tm);

private:

    /* member variables */

    /* Start of measurement */
#ifdef _WIN32
    LARGE_INTEGER start_time;
    LARGE_INTEGER end_time;
#else
    struct timeval start_time;
    struct timeval t_time;
#endif

    /* Time difference between the last start and stop */
    float diff_time;

    /* TOTAL time difference between starts and stops */
    float total_time;

    /* flag if the stop watch is running */
    bool running;

    /* Number of times clock has been started */
    /* and stopped to allow averaging */
    int clock_sessions;

#ifdef _WIN32
    double freq;
    bool freq_set;
#endif
};

inline StopWatch & StopWatch::operator=(const float tm)
{
    total_time = tm;
    diff_time = tm;
    clock_sessions = 1;
    return *this;
}

inline StopWatch & StopWatch::operator+=(const float tm)
{
    diff_time = tm;
    total_time += diff_time;
    clock_sessions++;
    return *this;
}

