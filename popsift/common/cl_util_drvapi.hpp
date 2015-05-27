#ifndef __CL_UTIL_DRVAPI_HPP__
#define __CL_UTIL_DRVAPI_HPP__

#include "cl_util.hpp"

/* CL library include */
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

/** 
 * print available platforms and device if in verbose mode        
 * create a cl context, choose device, and return platform id     
 * 
 * @param deviceType 
 * @param platformId 
 * @param verbose 
 * @param choose 
 * 
 * @return 
 */
inline cl_context _create_context(cl_device_type * deviceType,
                                  char *platformId, bool verbose, bool choose)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id *platforms;
    cl_platform_id firstPlatformId = NULL;
    cl_context context = NULL;
    cl_uint numDevices = 0;
    cl_device_id *devices;
    char cBuffer[1024];

    cl_uint i;
    int preferredPlatform = -1, preferredDevice = -1;

    if (choose)
        verbose = true;

    /* First, select an OpenCL platform to run on.  For this example, we */
    /* simply choose the first available platform.  Normally, you would */
    /* query for all available platforms and select the most appropriate one. */
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (errNum != CL_SUCCESS) {
        std::cerr << "ERROR: " << "clGetPlatformIDs failed" << std::endl;
        return NULL;
    }

    platforms = new cl_platform_id[sizeof(cl_platform_id) * numPlatforms];
    errNum = clGetPlatformIDs(numPlatforms, platforms, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0) {
        std::cerr << "ERROR: " << "Failed to find any OpenCL platforms." << std::endl;
        return NULL;
    }

    /* using default platform */
    if (platformId != NULL && !choose) {
        for (i = 0; i < numPlatforms; ++i) {
            clGetPlatformInfo
                (platforms[i], CL_PLATFORM_VENDOR, sizeof(cBuffer),
                 &cBuffer, NULL);
            if (!strcmp(cBuffer, platformId)) {
                firstPlatformId = platforms[i];
                break;
            } else if (!strcmp(cBuffer, platformId)) {
                firstPlatformId = platforms[i];
                break;
            } else if (!strcmp(cBuffer, platformId)) {
                firstPlatformId = platforms[i];
                break;
            } else
                continue;
        }
        if (firstPlatformId == NULL) {
            std::cerr << "ERROR: " << "No " << platformId 
                      << " platform available for your device(s)." << std::endl;
            return NULL;
        }
    }
    /* Print and let the user choose which platform */
    else {
        /* initialize platformid with a random platform -- usually AMD comes first */
        for (i = 0; i < numPlatforms; ++i) {
            clGetPlatformInfo
                (platforms[i], CL_PLATFORM_VENDOR, sizeof(cBuffer),
                 &cBuffer, NULL);
            if (!strcmp(cBuffer, "Advanced Micro Devices, Inc.")
                && *deviceType == CL_DEVICE_TYPE_CPU) {
                firstPlatformId = platforms[i];
                break;
            } else if (!strcmp(cBuffer, "NVIDIA Corporation") &&
                       *deviceType == CL_DEVICE_TYPE_GPU) {
                firstPlatformId = platforms[i];
                break;
            } else if (!strcmp(cBuffer, "Intel(R) Corporation") &&
                       *deviceType == CL_DEVICE_TYPE_CPU) {
                firstPlatformId = platforms[i];
                break;
            } else
                continue;
        }

        verbose && std::cerr << " ---------------------------------" << std::endl;
        verbose && std::cerr << " Number of Platforms = " << numPlatforms << std::endl;
        choose  && std::cerr << " Please choose which platform you would like to use " << std::endl;
        for (i = 0; i < numPlatforms; ++i) {
            clGetPlatformInfo
                (platforms[i], CL_PLATFORM_VENDOR, sizeof(cBuffer),
                 &cBuffer, NULL);
            verbose && std::cerr << " Platform [" << i << "]: " << cBuffer << std::endl;
        }

        if (choose) {
            std::cerr << " Insert a number: ";
            std::cin >> preferredPlatform;
            if (preferredPlatform >= 0
                && (cl_uint) preferredPlatform < numPlatforms) {
                firstPlatformId = platforms[preferredPlatform];
                *deviceType = CL_DEVICE_TYPE_ALL;
            } else {
                std::cerr << "ERROR: " << "There is no such platform" << std::endl;
                return NULL;
            }
        }
    }

    if (firstPlatformId == NULL) {
        std::cerr << "ERROR: " << "No platforms available for your device(s)." << std::endl;
        return NULL;
    }

    errNum = clGetDeviceIDs(firstPlatformId, *deviceType, 0, NULL, &numDevices);
    if (errNum != CL_SUCCESS) {
        if (*deviceType == CL_DEVICE_TYPE_GPU) {
            std::cerr << "ERROR: " << "No OpenCL GPU device for platform: " << platformId << std::endl;
        } else if (*deviceType == CL_DEVICE_TYPE_CPU) {
            std::cerr << "ERROR: " << "No OpenCL CPU device for platform: " << platformId << std::endl;
        }
        return NULL;
    }

    devices = new cl_device_id[sizeof(cl_device_id) * numDevices];
    errNum = clGetDeviceIDs
        (firstPlatformId, *deviceType, numDevices, devices, &numDevices);
    if (errNum != CL_SUCCESS || numDevices <= 0) {
        std::cerr << "ERROR: " << "Failed to find any OpenCL devices." << std::endl;
        return NULL;
    }

    /* Print all availables Devices */
    verbose && std::cerr << " ---------------------------------" << std::endl;
    verbose && std::cerr << " Number of Devices = " << numDevices << std::endl;
    choose && std::cerr << " Please choose which device you would like to use " <<      std::endl;
    for (i = 0; i < numDevices; ++i) {
        clGetDeviceInfo
            (devices[i], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
        verbose && std::cerr << " Device [" << i << "]: " << cBuffer << 
            std::endl;
    }
    verbose && std::cerr << " ---------------------------------" << std::endl;

    if (choose) {
        std::cerr << " Insert a number: ";
        std::cin >> preferredDevice;
        if (preferredDevice >= 0 && (cl_uint) preferredDevice < numDevices) {
            clGetDeviceInfo
                (devices[preferredDevice], CL_DEVICE_TYPE,
                 sizeof(cl_device_type), (void *) &deviceType[0], NULL);
        } else {
            std::cerr << "ERROR: " << "There is no such device" << std::endl;
            return NULL;
        }
    }

    delete[]devices;
    devices = NULL;

    devices = new cl_device_id[sizeof(cl_device_id) * numDevices];
    errNum = clGetDeviceIDs
        (firstPlatformId, *deviceType, numDevices, devices, &numDevices);
    if (errNum != CL_SUCCESS || numDevices <= 0) {
        std::cerr << "ERROR: " << "clGetDeviceIDs failed" << std::endl;
        return NULL;
    }

    /* Next, create an OpenCL context on the platform.  Attempt to */
    /* create a GPU-based context, and if that fails, try to create */
    /* a CPU-based context. */
    cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) firstPlatformId,
        0
    };

    /* Make a context based on the Device ID */
    context = clCreateContext
        (contextProperties, 1, devices, NULL, NULL, &errNum);
    if (errNum != CL_SUCCESS) {
        if (*deviceType == CL_DEVICE_TYPE_GPU) {
            std::cerr << "ERROR: " << "Could not create GPU context: " << 
                std::string(error_string(errNum)) << std::endl;
        }
        if (*deviceType == CL_DEVICE_TYPE_CPU) {
            std::cerr << "ERROR: " << "Could not create CPU context: " << 
                std::string(error_string(errNum)) << std::endl;
        }
        return NULL;
    }

    clGetPlatformInfo
        (firstPlatformId, CL_PLATFORM_VENDOR, sizeof(cBuffer), &cBuffer,
         NULL);

    if (platformId != NULL)
        strcpy(platformId, cBuffer);

    delete[]platforms;
    delete[]devices;
    return context;
}

/** 
 * Create and OpenCL program from the kernel source file
 * 
 * @param context 
 * @param device 
 * @param fileName 
 * @param compileOptions 
 * 
 * @return cl_program
 */
inline cl_program _create_program(cl_context context, cl_device_id device, const char *fileName,
                                  const char *compileOptions) 
{
    cl_int errNum;
    cl_program program;

    std::ifstream kernelFile(fileName, std::ios::in | std::ios::binary);
    if (!kernelFile.is_open()) {
        std::cerr << "ERROR: " << "Failed to open file for reading: " << fileName << std::endl;
        return NULL;
    }

    std::ostringstream oss;
    oss << kernelFile.rdbuf();

    std::string srcStdStr = oss.str();
    const char *srcStr = srcStdStr.c_str();
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) &srcStr,
                                        NULL, NULL);
    if (program == NULL) {
        std::cerr << "ERROR: " << "Failed to create CL program from source." << 
            std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, compileOptions, NULL, NULL);
    if (errNum != CL_SUCCESS) {

        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "ERROR: " << "Error in kernel: " << errNum << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

/* @brief: variety of calls */

inline cl_context create_context(cl_device_type & deviceType,
                                 std::string & platformId, bool verbose,
                                 bool choose)
{
    cl_context context = NULL;
    char *pi = NULL;
    bool v = verbose;
    bool c = choose;
    if (platformId != "") {
        pi = (char *) malloc(1024);
        strcpy(pi, platformId.c_str());
    }
    context = _create_context(&deviceType, pi, v, c);
    if (pi) {
        platformId = pi;
        free(pi);
    }
    return context;
}

inline cl_context create_context(cl_device_type & deviceType, bool verbose,
                                 bool choose)
{
    cl_context context = NULL;
    char *pi = NULL;
    bool v = verbose;
    bool c = choose;
    context = _create_context(&deviceType, pi, v, c);
    return context;
}

inline cl_context create_context(cl_device_type & deviceType)
{
    cl_context context = NULL;
    char *pi = NULL;
    bool v = false;
    bool c = false;
    context = _create_context(&deviceType, pi, v, c);
    return context;
}


/* @brief variety of calls */
inline cl_program create_program(cl_context context, cl_device_id device, const char *fileName,
                                 const char *compileOptions) 
{
    cl_program program = NULL;
    program = _create_program(context, device, fileName, compileOptions);
    return program;
}

inline cl_program create_program(cl_context context, cl_device_id device, const char *fileName) 
{
    cl_program program = NULL;
    char *c = NULL;
    program = _create_program(context, device, fileName, c);
    return program;
}

#endif
// end - __CL_UTIL_DRVAPI_HPP__
