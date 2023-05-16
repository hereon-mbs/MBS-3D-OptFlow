
#ifndef HDCOMMUNICATION_H
#define HDCOMMUNICATION_H

#include <vector>
#include <string.h>
#include <iostream>
#include <tiffio.h>
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>

namespace hdcom
{
    class HdCommunication
    {
    public:
    	short last_bps = 0;

    	/*************** Reading and saving unknown dimensionality ***************/
	std::vector<std::string> GetFilelist_And_ImageSequenceDimensions(std::string path, int outshape[3], bool &is_rgb);
    	float* GetTif_unknowndim_32bit(std::string inpath, int outshape[3], bool dspprogress=false);
    	float* GetTif_unknowndim_32bit(std::string path, int outshape[3], std::pair<int,int> zrange, bool dspprogress);
    	void SaveTif_unknowndim_32bit(float *image, int imgshape[3], std::string path, std::string name, std::string subdir3D = "", int slice_nr = -1);

    	/*************** Read ImageJ 3D-tif and/or read without libtiff ***************/
    	float* Custom3DTifReader(std::string path, int outshape[3], bool verbose = false);

        /*************** Reading a single greyscale image as 1D vector ***************/
        std::vector<float> Get2DTifImage_32bit(std::string path, int outshape[2]);
        float* Get2DTifImage_32bitPointer(std::string file, int outshape[2]);
        void Insert2DTifImage_32bitPointer(std::string file, int outshape[2], float *imgpointer, long long int pos0);
        std::vector<uint8_t> Get2DTifImage_8bit(std::string file, int outshape[2]);

        /**************** Reading a greyscale image sequence (with overloading) ****************/
        float* Get3DTifSequence_32bitPointer(std::vector<std::string> &filelist, int outshape[3], bool dspprogress);
        std::vector<float> Get3DTifSequence_32bit(std::string path, int outshape[3]);
        std::vector<float> Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3], bool dspprogress);
        std::vector<float> Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3]);
        std::vector<uint8_t> Get3DTifSequence_8bit(std::string path, int outshape[3]);
        std::vector<uint8_t> Get3DTifSequence_8bit(std::vector<std::string> &filelist, int outshape[3]);

        /********************** Saving a single greyscale image **********************/
        void Save2DTifImage_8bit(uint8_t *image, int imgshape[2], std::string path, std::string name);
        void Save2DTifImage_32bit(int *image, int imgshape[2], std::string path, std::string name, int64_t pos);
        void Save2DTifImage_8bit(std::vector<uint8_t> &image, int imgshape[2], std::string path, std::string name, int64_t pos);
        void Save2DTifImage_32bit(int *image, int imgshape[2], std::string path, std::string name);
        void Save2DTifImage_32bit(std::vector<float> &image, int imgshape[2], std::string path, std::string name, int64_t pos);
        void Save2DTifImage_32bit(float *image, int imgshape[2], std::string path, std::string name, int64_t pos);
        void Save2DTifImage_RGB(uint8_t *image, int imgshape[2], std::string path, std::string name);

        /********************** Saving a greyscale sequence **********************/
        void SaveTifSequence_32bit(std::vector<float> &image, int imgshape[3], std::string path, std::string name, bool dspprogress);
        void SaveTifSequence_32bit(float *image, int imgshape[3], std::string path, std::string name, bool dspprogress);
        void SaveTifSequence_32bit(int *image, int imgshape[3], std::string path, std::string name);
        void SaveTifSequence_8bit(std::vector<uint8_t> &image, int imgshape[3], std::string path, std::string name, int firstslice);

        /********************** Saving a vector **********************/
        void Save3DVector_vtk(float *u, int shape[3], std::string path, std::string name, std::string header="");

        //Get a vector with tif-files in a directory:
        int GetFilelist(std::string const directory, std::vector<std::string> &outfiles);
        std::vector<std::string> GetFilelist(std::string path, int outshape[3]);
        void makedir(const std::string path);
	bool path_exists(const std::string& path);
	bool is_absolute_path(const std::string& path);

    private:
        bool hasEnding(std::string const &str1, std::string const &str2);

    };
}

#endif // HDCOMMUNICATION_H
