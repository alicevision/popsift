/*
 * Copyright 2018, Simula Research Laboratory
 * Copyright 2018, University of Oslo
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sys/stat.h> // for stat
#ifdef _WIN32
#include <direct.h>
#define stat _stat
#define mkdir(path, perm) _mkdir(path)
#endif

#include "sift_task_extract.h"
#include "sift_job.h"
#include "popsift.h"
#include "sift_features.h"

using namespace std;

/*********************************************************************************
 * TaskExtract
 *********************************************************************************/

void TaskExtract::loop( )
{
    extractDownloadLoop();
}

void TaskExtract::extractDownloadLoop( )
{
    SiftJob* job;
    while( ( job = _op->getNextJob() ) != 0 )
    {
        popsift::FeaturesHost* features;

        popsift::ImageBase* img = job->getImg();
        _op->uploadImageFromJob( img );
        _op->returnImageToPool( img );
        _op->findKeypoints( );
        features = _op->downloadFeaturesToHost( );

        logDescriptors( features );

        job->setFeatures( features );
    }
}

SiftJob* TaskExtract::newJob( int w, int h, const unsigned char* imageData )
{
    return new SiftJob( w, h, imageData );
}

SiftJob* TaskExtract::newJob( int w, int h, const float* imageData )
{
    return new SiftJob( w, h, imageData );
}

void TaskExtract::logDescriptors( popsift::FeaturesHost* features )
{
    bool log_to_file = ( _config.getLogMode() == popsift::Config::All );

    if( not log_to_file ) return;

    const char* basename = "pyramid";

    _op->logPyramid( basename );

    struct stat st = { 0 };
    if (stat("dir-desc", &st) == -1) {
        mkdir("dir-desc", 0700);
    }
    ostringstream ostr;
    ostr << "dir-desc/desc-" << basename << ".txt";
    ofstream of(ostr.str().c_str());
    writeDescriptors( of, features, true, true );

    if (stat("dir-fpt", &st) == -1) {
        mkdir("dir-fpt", 0700);
    }
    ostringstream ostr2;
    ostr2 << "dir-fpt/desc-" << basename << ".txt";
    ofstream of2(ostr2.str().c_str());
    writeDescriptors( of2, features, false, true );
}

void TaskExtract::writeDescriptors( ostream& ostr, popsift::FeaturesHost* features, bool really, bool with_orientation )
{
    int num_keypoints = features->getFeatureCount();

    if( num_keypoints == 0 ) return;

    const float up_fac = _config.getUpscaleFactor();

    for( int kpt_idx = 0; kpt_idx < num_keypoints; kpt_idx++ ) {
        const popsift::Feature& ext = features->getFeature( kpt_idx );

        const int   octave  = ext.octave; 
        const float xpos    = ext.xpos  * pow(2.0f, octave - up_fac);
        const float ypos    = ext.ypos  * pow(2.0f, octave - up_fac);
        const float scale   = ext.scale * pow(2.0f, octave - up_fac);
        for( int ori = 0; ori<ext.num_ori; ori++ ) {
            // const int   ori_idx = ext.idx_ori + ori;
            float       dom_ori = ext.orientation[ori];

            dom_ori = dom_ori / M_PI2 * 360;
            if (dom_ori < 0) dom_ori += 360;

            const popsift::Descriptor& desc  = *ext.desc[ori]; // hbuf.desc[ori_idx];

            if( with_orientation )
                ostr << setprecision(5)
                     << xpos << " "
                     << ypos << " "
                     << scale << " "
                     << dom_ori << " ";
            else
                ostr << setprecision(5)
                     << xpos << " " << ypos << " "
                     << 1.0f / (scale * scale)
                     << " 0 "
                     << 1.0f / (scale * scale) << " ";

            if (really) {
                for (int i = 0; i<128; i++) {
                    ostr << desc.features[i] << " ";
                }
            }
            ostr << endl;
        }
    }
}

