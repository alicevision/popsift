/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stack>
#include <queue>
#include <future>
#include <boost/thread/thread.hpp>
#include <boost/thread/sync_queue.hpp>

#include "sift_conf.h"
#include "sift_extremum.h"
#include "sift_job.h"
#include "sift_task.h"
#include "common/plane_2d.h"


#ifdef USE_NVTX
#include <nvToolsExtCuda.h>
#else
#define nvtxRangeStartA(a)
#define nvtxRangeEnd(a)
#endif

/* user parameters */
namespace popsift
{
    class ImageBase;
    class Pyramid;
    class FeaturesBase;
    class FeaturesHost;
    class FeaturesDev;

}; // namespace popsift

class PopSift;

/*********************************************************************************
 * PopSift
 *********************************************************************************/

class PopSift
{
    struct Pipe
    {
        boost::thread*                         _thread_stage1;
        boost::thread*                         _thread_stage2;
        boost::sync_queue<SiftJob*>            _queue_stage1;
        boost::sync_queue<SiftJob*>            _queue_stage2;
        boost::sync_queue<popsift::ImageBase*> _unused;
        popsift::ImageBase*                    _current;

        popsift::Pyramid*                      _pyramid;
    };

public:
    enum ImageMode
    {
        ByteImages,
        FloatImages
    };

public:
    /* We support more than 1 streams, but we support only one sigma and one
     * level parameters.
     */
    PopSift( ImageMode imode = ByteImages );
    PopSift( const popsift::Config&          config,
             Task*                           task,
             // popsift::Config::ProcessingMode mode  = popsift::Config::ExtractingMode,
             ImageMode                       imode = ByteImages );
    ~PopSift();

public:
    /** Provide the configuration if you used the PopSift default
     *  constructor */
    bool configure( const popsift::Config& config, bool force = false );

    void uninit( );

    /** Enqueue a byte image,  value range 0..255 */
    SiftJob*  enqueue( int                  w,
                       int                  h,
                       const unsigned char* imageData );

    /** Enqueue a float image,  value range 0..1 */
    SiftJob*  enqueue( int          w,
                       int          h,
                       const float* imageData );

    /** deprecated */
    inline void uninit( int /*pipe*/ ) { uninit(); }

    /** deprecated */
    inline bool init( int /*pipe*/, int w, int h ) {
        _last_init_w = w;
        _last_init_h = h;
        return true;
    }

    /** deprecated */
    inline popsift::FeaturesBase* execute( int /*pipe*/, const unsigned char* imageData )
    {
        SiftJob* j = enqueue( _last_init_w, _last_init_h, imageData );
        if( !j ) return 0;
        popsift::FeaturesBase* f = j->getBase();
        delete j;
        return f;
    }

    /** Get next job from second level queue.
     *  Called by a Task loop.
     */
    SiftJob* getNextJob( );

    /** Uploading an image into the GPU.
     *  Called by a Task loop.
     */
    void uploadImageFromJob( popsift::ImageBase* img );

    /** Return GPU-sided memory for image to the pool.
     *  Called by a Task loop.
     */
    void returnImageToPool( popsift::ImageBase* img );

    /** Run main SIFT algorithm to completion.
     *  Called by a Task loop.
     */
    void findKeypoints( );

    /** Allocate host-side memory dynamically and copy keypoints and
     *  descriptors from device.
     *  Called by a Task loop.
     */
    popsift::FeaturesHost* downloadFeaturesToHost( );

    /** Allocate device-side memory dynamically and copy keypoints and
     *  descriptors from device.
     *  Called by a Task loop.
     */
    popsift::FeaturesDev* cloneFeaturesOnDevice( );

    /** Allocate device-side memory dynamically and copy an entire layer
     *  from the pyramid.
     *  Called by a Task loop.
     */
    popsift::Plane2D<float>* cloneLayerOnDevice( int octave, int level );

    /** Get current number of octaves for pipe.
     *  Called by a Task loop.
     */
    int getNumOctaves( );

    /** Get current number of levels for pipe.
     *  Called by a Task loop.
     */
    int getNumLevels( );

    /** Take all layers of the Pyramid and the DoG pyramid and write
     *  them to disk.
     *  Called by a Task loop.
     */
    void logPyramid( const char* basename );

private:
    bool private_init( int w, int h );
    void uploadImages( );

private:
    Pipe             _pipe;
    popsift::Config  _config;

    /* Keep a copy of the config to avoid unnecessary re-configurations
     * in configure()
     */
    popsift::Config  _shadow_config;

    int              _last_init_w; /* to support depreacted interface */
    int              _last_init_h; /* to support depreacted interface */
    ImageMode        _image_mode;

private:
    /* Different operating modes are hidden behind the Task class.
     * The processing mode will eventually become irrelevant.
     */
    /* The tasks refer to alternative worker functions for Jobs submitted by
     * a calling application. The choice of method is made by the mode parameter
     * in the PopSift constructor. */
    Task* _task;

    // popsift::Config::ProcessingMode  _proc_mode;
};

