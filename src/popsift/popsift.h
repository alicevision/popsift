/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include "common/sync_queue.h"
#include "common/device_prop.h"
#include "sift_conf.h"
#include "sift_config.h"
#include "sift_extremum.h"

#include <cuda_runtime.h>

#include <future>
#include <queue>
#include <stack>
#include <thread>
#include <vector>

#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
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

class SiftJob
{
    std::promise<popsift::FeaturesBase*> _p;
    std::future <popsift::FeaturesBase*> _f;
    int                 _w;
    int                 _h;
    unsigned char*      _imageData;
    popsift::ImageBase* _img;
#if POPSIFT_IS_DEFINED(POPSIFT_USE_NVTX)
    nvtxRangeId_t       _nvtx_id;
#endif

public:

    /**
     * @brief Constructor for byte images, value range 0..255
     * @param[in] w the width in pixel of the image
     * @param[in] h the height in pixel of the image
     * @param[in] imageData the image buffer
     */
    SiftJob( int w, int h, const unsigned char* imageData );

    /**
     * @brief Constructor for float images, value range [0..1[
     * @param[in] w the width in pixel of the image
     * @param[in] h the height in pixel of the image
     * @param[in] imageData the image buffer
     */
    SiftJob( int w, int h, const float* imageData );

    /**
     * @brief Destructor releases all the resources.
     */
    ~SiftJob( );

    /**
     * @deprecated
     * @see getHost()
     */
    popsift::FeaturesHost* get();
    popsift::FeaturesBase* getBase();
    /**
     * @brief
     * @return
     */
    popsift::FeaturesHost* getHost();
    popsift::FeaturesDev*  getDev();

    void setImg( popsift::ImageBase* img );
    popsift::ImageBase* getImg();

    /** fulfill the promise */
    void setFeatures( popsift::FeaturesBase* f );
};

/**
 * @brief
 */
class PopSift
{
    struct Pipe
    {
        std::unique_ptr<std::thread>            _thread_stage1;
        std::unique_ptr<std::thread>            _thread_stage2;
        popsift::SyncQueue<SiftJob*>            _queue_stage1;
        popsift::SyncQueue<SiftJob*>            _queue_stage2;
        popsift::SyncQueue<popsift::ImageBase*> _unused;

        popsift::Pyramid*                      _pyramid{nullptr};

        /**
         * @brief Release the allocated resources, if any.
         */
        void uninit();
    };

public:

    /**
    * @brief Image modes
    */
    enum ImageMode
    {
        ///  byte image, value range 0..255
        ByteImages,
        /// float images, value range [0..1[
        FloatImages
    };

    /**
     * @brief Results for the allocation test.
     */
    enum AllocTest
    {
        /// the image dimensions are supported by this device's CUDA texture engine.
        Ok,
        /// the input image size exceeds the dimensions of the CUDA Texture used for loading.
        ImageExceedsLinearTextureLimit,
        /// the scaled input image exceeds the dimensions of the CUDA Surface used for the image pyramid.
        ImageExceedsLayeredSurfaceLimit
    };

public:

    PopSift() = delete;
    PopSift(const PopSift&) = delete;

    /**
     * @brief We support more than 1 streams, but we support only one sigma and one
     * level parameters.
     */
    explicit PopSift( ImageMode imode = ByteImages, int device = 0 );

    /**
     * @brief
     * @param config
     * @param mode
     * @param imode
     */
    explicit PopSift(const popsift::Config& config,
                     popsift::Config::ProcessingMode mode = popsift::Config::ExtractingMode,
                     ImageMode imode = ByteImages, int device = 0);

    /**
     * @brief Release all the resources.
     */
    ~PopSift();

public:
    /**
     * @brief Provide the configuration if you used the PopSift default
     *  constructor
     */
    bool configure( const popsift::Config& config, bool force = false );

    /**
     * @brief Release the resources.
     */
    void uninit( );

    /**
     *  @brief Check whether the current CUDA device can support the image
     *  resolution (width,height) with the current configuration
     *  based on the card's texture engine.
     *  The function does not check if there is sufficient available
     *  memory.
     *
     *  The first part of the test depends on the parameters width and
     *  height. It checks whether the image size is supported by CUDA
     *  2D linear textures on this card. This is used to load the image
     *  into the first level of the first octave.
     *  For the second part of the tst, two value of the configuration
     *  are important: 
     *  "downsampling", because it determines the required texture size
     *  after loading. The CUDA 2D layered texture must support the
     *  scaled width and height.
     *  "levels", because it determines the number of levels in each
     *  octave. The CUDA 2D layered texture must support enough depth
     *  for each level.
     *
     * @param[in] width  The width of the input image
     * @param[in] height The height of the input image
     * @return AllocTest::Ok if the image dimensions are supported by this device's
     *         CUDA texture engine,
     *         AllocTest::ImageExceedsLinearTextureLimit if the input image size
     *         exceeds the dimensions of the CUDA Texture used for loading.
     *         The input image must be scaled.
     *         AllocTest::ImageExceedsLayeredSurfaceLimit if the scaled input
     *         image exceeds the dimensions of the CUDA Surface used for the
     *         image pyramid. The scaling factor must be changes to fit in.
     * @remark  * If you want to call configure() before extracting features,
     *           you should call configure() before textTextureFit().
     * @remark  * The current CUDA device is determined by a call to
     *           cudaGetDevice(), card properties are only read once.
     * @see AllocTest
     */
    AllocTest testTextureFit( int width, int height );

    /**
     * @brief Create a warning string for an AllocTest error code.
     */
    std::string testTextureFitErrorString( AllocTest err, int w, int h );

    /**
     * @brief Enqueue a byte image,  value range [0,255].
     * @param[in] w the width of the image.
     * @param[in] h the height of the image.
     * @param[in] imageData the image buffer.
     * @return the associated job
     * @see SiftJob
     */
    SiftJob*  enqueue( int                  w,
                       int                  h,
                       const unsigned char* imageData );

    /**
     * @brief Enqueue a float image,  value range [0,1].
     * @param[in] w the width of the image.
     * @param[in] h the height of the image.
     * @param[in] imageData the image buffer.
     * @return the associated job
     * @see SiftJob
     */
    SiftJob*  enqueue( int          w,
                       int          h,
                       const float* imageData );

    /**
     * @deprecated
     */
    inline void uninit( int /*pipe*/ ) { uninit(); }

    /**
     * @deprecated
     */
    inline bool init( int /*pipe*/, int w, int h ) {
        _last_init_w = w;
        _last_init_h = h;
        return true;
    }

    /**
     * @deprecated
     */
    inline popsift::FeaturesBase* execute( int /*pipe*/, const unsigned char* imageData )
    {
        SiftJob* j = enqueue( _last_init_w, _last_init_h, imageData );
        if( !j ) return nullptr;
        popsift::FeaturesBase* f = j->getBase();
        delete j;
        return f;
    }

private:
    bool applyConfiguration( bool force = false );

    bool private_init( int w, int h );
    bool private_uninit( );
    void private_apply_scale_factor( int& w, int& h );
    void uploadImages( );

    /* The following methods are alternative worker functions for Jobs submitted by
     * a calling application. The choice of method is made by the mode parameter
     * in the PopSift constructor. */

    /* Worker function: Extract SIFT features and download to host */
    void extractDownloadLoop( );

    /* Worker function: Extract SIFT features, clone results in device memory */
    void matchPrepareLoop( );

private:
    Pipe            _pipe;
    popsift::Config _config;

    /* Keep a copy of the config to avoid unnecessary re-configurations
     * in configure()
     */
    popsift::Config _shadow_config;

    int             _last_init_w{}; /* to support deprecated interface */
    int             _last_init_h{}; /* to support deprecated interface */
    ImageMode       _image_mode;
    int             _device;

    /// whether the object is initialized
    bool            _isInit{true};

    // Device property collection runs when this object is created
    popsift::cuda::device_prop_t   _device_properties;
};

