#include "hdcommunication.h"
#include "auxiliary.h"
#include <arpa/inet.h>

namespace hdcom
{
    using namespace std;

    void DummyHandler(const char* module, const char* fmt, va_list ap){
    // ignore errors and warnings (or handle them your own way)
    }
    int makepath(std::string s)
    {
        size_t pos=0;
        std::string dir;
        int mdret;

        if(s[s.size()-1]!='/'){s+='/';}

        while((pos=s.find_first_of('/',pos))!=std::string::npos){
            dir=s.substr(0,pos++);
            if(dir.size()==0) continue; // if leading / first time is 0 length
            if((mdret=mkdir(dir.c_str(),0777)) && errno!=EEXIST){
                return mdret;
            }
        }
        return mdret;
    }
    bool file_exists(const std::string& filename)
    {
    	struct stat buffer;
    	return (stat (filename.c_str(), &buffer) == 0);
    }
    bool HdCommunication::path_exists(const std::string& path)
    {
        struct stat info;
        if (stat(path.c_str(), &info) != 0)
        {
            return false;
        }
        return (info.st_mode & S_IFDIR) != 0;
    }
    bool HdCommunication::is_absolute_path(const std::string& path)
    {
        bool absolute_path = false;
	std::string rootpath = path.substr(0, path.rfind("/", path.length()-2)+1);

        while(!absolute_path)
        {
            if(!path_exists(rootpath)) rootpath = rootpath.substr(0, rootpath.rfind("/", rootpath.length()-2)+1);
	    else return true;

	    if (rootpath.length() < 2)
		break;
        }
        return absolute_path;
    }

	/*************** Reading unknown dimensionality ***************/
	std::vector<std::string> HdCommunication::GetFilelist_And_ImageSequenceDimensions(std::string path, int outshape[3], bool &is_rgb)
	{
		std::vector<std::string> filelist;

		if(hasEnding(path, "tif") || hasEnding(path, "tiff"))
		{
			filelist.push_back(path);

			if(!file_exists(path)) filelist[0] = "missing";
			return filelist;
		}
		else
		{
			if(!path_exists(path))
			{
				filelist.push_back("missing");
				return filelist;
			}

			GetFilelist(path+"/", filelist);

			if (filelist.size() == 0)
			{
			    filelist.push_back("no tif");
			    return filelist;
			}
		}

		char *inpath;
		inpath = new char[filelist[0].length()+1];
		strcpy(inpath, filelist[0].c_str());

		TIFFSetWarningHandler(DummyHandler);
		TIFF* tif = TIFFOpen(inpath, "r");
		short nsamples;

		//check image size
		if (tif)
		{
			 TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &outshape[1]); //dimension1
			 TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &outshape[0]); //dimension0
			 TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
			 outshape[2] = filelist.size();

			 //std::cout << nsamples << std::endl;

			 is_rgb = (nsamples == 3 || nsamples == 4) ? true : false;
		}
		else
			std::cout << "Error! not a tif-sequence!" << std::endl;

		TIFFClose(tif);

		return filelist;
	}
	float* HdCommunication::GetTif_unknowndim_32bit(std::string path, int outshape[3], bool dspprogress)
	{
		if(hasEnding(path, "tif") || hasEnding(path, "tiff"))
		{
			outshape[2] = 1;
			float* output = Custom3DTifReader(path, outshape, false);

			if (dspprogress)
				printf("Finished reading stack of %u x %u x %u voxels \033[1;33m(experimental I/O)\033[0m\n",outshape[0],outshape[1],outshape[2]);

			return output;
			//return Get2DTifImage_32bitPointer(path, outshape);
		}
		else
		{
			vector<string> filelist;
			GetFilelist(path+"/", filelist);
			return Get3DTifSequence_32bitPointer(filelist, outshape, dspprogress);
		}
	}
	float* HdCommunication::GetTif_unknowndim_32bit(std::string path, int outshape[3], std::pair<int,int> zrange, bool dspprogress)
	{
		if(hasEnding(path, "tif") || hasEnding(path, "tiff"))
		{
			outshape[2] = 1;
			float* output = Custom3DTifReader(path, outshape, false);

			//unnecessary to read in all and crop but probably not going to be used anyways
			if (zrange.first >= 0 && zrange.first < outshape[2] && zrange.second >= 0 && zrange.second < outshape[2])
			{
				outshape[2] = zrange.second-zrange.first;
				long long int nslice = outshape[0]*outshape[1];
				long long int nstack = outshape[2]*nslice;

				float *cropped_output = (float*) malloc(nstack*sizeof(*cropped_output));

				#pragma omp parallel for
				for (long long int idx = 0; idx < nstack; idx++)
					cropped_output[idx] = output[idx+zrange.first*nslice];

				free(output);
				std::swap(output, cropped_output);
				return cropped_output;
			}


			if (dspprogress)
				printf("Finished reading stack of %u x %u x %u voxels \033[1;33m(experimental I/O)\033[0m\n",outshape[0],outshape[1],outshape[2]);

			return output;
			//return Get2DTifImage_32bitPointer(path, outshape);
		}
		else
		{
			vector<string> filelist;
			GetFilelist(path+"/", filelist);

			if (zrange.second > 0 && zrange.second <= filelist.size())
			{
				vector<string> sublist(filelist.begin(), filelist.begin()+zrange.second);
				filelist = sublist;
			}
			if (zrange.first > 0 && zrange.first < filelist.size())
			{
				vector<string> sublist(filelist.begin()+zrange.first, filelist.end());
				filelist = sublist;
			}

			return Get3DTifSequence_32bitPointer(filelist, outshape, dspprogress);
		}
	}
	void HdCommunication::SaveTif_unknowndim_32bit(float *image, int imgshape[3], std::string path, std::string name, std::string subdir3D, int slice_nr)
	{
		if (imgshape[2] <= 1)
			Save2DTifImage_32bit(image, imgshape, path, name, 0);
		else if (slice_nr != -1)
		{
			int64_t pos = imgshape[0]*imgshape[1];
			pos *= slice_nr;

			Save2DTifImage_32bit(image, imgshape, path, name + aux::zfill_int2string(slice_nr,4), pos);
		}
		else
			SaveTifSequence_32bit(image, imgshape, path+"/"+subdir3D+"/", name, false);

		return;
	}

	/*************** Read ImageJ 3D-tif and/or read without libtiff ***************/
	float* HdCommunication::Custom3DTifReader(std::string path, int outshape[3], bool verbose)
	{
		/*
		 * Tested with 3D-Tiffs saved with ImageJ.
		 * Assumes that there is only one IFD
		 *
		 */

		outshape[0] = 0;
		outshape[1] = 0;
		outshape[2] = 1;

		int bps = 1;
		bool floating_point = false;
		int offset270 = 0; //image description offset
		bool is_bigendian = false;
		std::streampos datapos = -1;

		ifstream tiffile (path.c_str(), ios::binary);

		//extract file size
		////////////////////////////////////////////////////
		std::streampos fsize = tiffile.tellg();
		tiffile.seekg(0, ios::end);
		fsize = tiffile.tellg() - fsize;
		tiffile.seekg(0,ios::beg);
		////////////////////////////////////////////////////

		//8 bytes image file header
		////////////////////////////////////////////////////
		//read byte order
		char buffer[2];
		tiffile.read(buffer, 2);
		if (buffer[0] != 0x49) is_bigendian = true;

		//and compare with endianness of machine
		if (htonl(47) == 47 && is_bigendian) is_bigendian = false;
		else if (htonl(47) == 47) is_bigendian = true; //data is little endian. need byte swap
		//else machine is little endian as assumed

		//read magic number 42 and IFD offset
		unsigned int ifd_offset;
		short magic_number;
		tiffile.read((char*)&magic_number, 2);
		tiffile.read((char*)&ifd_offset, 4);

		if (is_bigendian){
			magic_number = ntohs(magic_number);
			ifd_offset = ntohl(ifd_offset);
		}

		if(verbose) std::cout << "ifd-offset: " << ifd_offset << std::endl;

		tiffile.seekg(ifd_offset,ios::beg);

		if (magic_number != 42)
			std::cout << "Warning! Magic number in tiff reader is not 42!" << std::endl;
		////////////////////////////////////////////////////

		//image file directory (each entry 12 bytes)
		////////////////////////////////////////////////////
		unsigned short n_tags;
		tiffile.read((char*)&n_tags, 2);
		if (is_bigendian) n_tags = ntohs(n_tags);

		if(verbose) cout << "n_tags: " << n_tags << endl;

		unsigned short tag_ID, tag_fieldtype, tag_offset;
		unsigned long tag_count, tag_value;

		for (int i = 0; i < n_tags; i++)
		{
			tiffile.read((char*)&tag_ID, 2);
			tiffile.read((char*)&tag_fieldtype, 2);

			if (is_bigendian)
			{
				tag_ID = ntohs(tag_ID);
				tag_fieldtype = ntohs(tag_fieldtype);
			}

			if(tag_fieldtype == 4)
			{

				tiffile.read((char*)&tag_count, 4);
				tiffile.read((char*)&tag_value, 4);

				if (is_bigendian){
					tag_count = ntohl(tag_count);
					tag_value = ntohl(tag_value);
				}
			}
			else if (tag_fieldtype == 3)
			{
				if(is_bigendian)
				{
					unsigned short tmp;
					tiffile.read((char*)&tag_count, 4);
					tiffile.read((char*)&tmp, 2);
					tiffile.read((char*)&tag_offset, 2);

					tag_count = ntohs(tag_count);
					tag_value = ntohs(tmp);
				}
				else
				{
					unsigned char temp[4];
					tiffile.read(reinterpret_cast<char*>(temp), 4);
					tag_count = reinterpret_cast<unsigned long&>(temp);
					tiffile.read(reinterpret_cast<char*>(temp), 2);
					tag_value = reinterpret_cast<unsigned short&>(temp);
					tiffile.read(reinterpret_cast<char*>(temp), 2);
				}
			}
			else
			{
				tiffile.read((char*)&tag_count, 4);
				tiffile.read((char*)&tag_value, 4);

				if (is_bigendian){
					tag_count = ntohl(tag_count);
					tag_value = ntohl(tag_value);
				}
			}

			if (tag_ID == 256) outshape[0] = tag_value;
			if (tag_ID == 257) outshape[1] = tag_value;
			if (tag_ID == 258) bps = tag_value;
			if (tag_ID == 273) datapos = (std::streampos) tag_value;
			if (tag_ID == 339 && tag_value == 3) floating_point = true;
			if (tag_ID == 270) offset270 = tag_count;

			if(verbose) std::cout << "tag " << tag_ID  << " (" << tag_fieldtype << "): " << tag_count << " "<< tag_value << std::endl;
		}
		////////////////////////////////////////////////////

		//estimate stack depth from image size
		////////////////////////////////////////////////////
		long long int nslice = outshape[0]*outshape[1];
		outshape[2] = ((int) fsize-offset270)/(nslice*bps/8);
		long long int nstack = nslice*outshape[2];
		////////////////////////////////////////////////////

		if(verbose) std::cout << "\ndimensions: " << outshape[0] << "x" << outshape[1] << "x" << outshape[2] << ", " << bps << " bps, " << floating_point << ", " << offset270 << endl;
		////////////////////////////////////////////////////

		//get next IFD
		////////////////////////////////////////////////////
		unsigned long next_ifd;
		tiffile.read((char*)&next_ifd, 4);
		if (is_bigendian) next_ifd = ntohl(next_ifd);
		////////////////////////////////////////////////////

		//get description
		////////////////////////////////////////////////////
		char description[offset270];
		tiffile.read(description, offset270);

		if(verbose)
		{
			std::cout << "\nimage description:\n----------------\n";
			for (int i = 0; i < offset270; i++) std::cout << description[i];
			std::cout << endl;
			std::cout << "----------------" << std::endl;
		}
		////////////////////////////////////////////////////

		float *output = (float*) malloc(nstack*sizeof(*output));

		//move to the beginning of the data stream
		////////////////////////////////////////////////////
		if (ifd_offset > 8) tiffile.seekg(8,ios::beg); //case where header is followed by data stream immediately (files saved with libtiff)
		else if (datapos != -1) tiffile.seekg(datapos, ios::beg); //given with tag273
		////////////////////////////////////////////////////

		//read 8bit tiff
		////////////////////////////////////////////////////
		if (bps == 8)
		{
			unsigned char *data = (unsigned char*) malloc((nstack)*sizeof(*data));
			tiffile.read(reinterpret_cast<char*>(data), (nstack));

			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++)
				output[pos] = (float) reinterpret_cast<uint8_t&>(data[pos]);

			free(data);
		}
		////////////////////////////////////////////////////

		//read 32bit floating point
		////////////////////////////////////////////////////
		else if (bps == 32 && floating_point)
		{
			unsigned char *data = (unsigned char*) malloc((4*nstack)*sizeof(*data));
			tiffile.read(reinterpret_cast<char*>(data), (4*nstack));

			if(!is_bigendian)
			{
				#pragma omp parallel for
				for (long long int pos = 0; pos < nstack; pos++)
				{
					unsigned char temp[4];
					temp[0] = data[4*pos];
					temp[1] = data[4*pos+1];
					temp[2] = data[4*pos+2];
					temp[3] = data[4*pos+3];
					output[pos] = reinterpret_cast<float&>(temp);
				}
			}
			else
			{
				#pragma omp parallel for
				for (long long int pos = 0; pos < nstack; pos++)
				{
					unsigned char temp[4];
					temp[3] = data[4*pos];
					temp[2] = data[4*pos+1];
					temp[1] = data[4*pos+2];
					temp[0] = data[4*pos+3];
					output[pos] = reinterpret_cast<float&>(temp);
				}
			}
			free(data);
		}
		////////////////////////////////////////////////////

		//read 16bit integer type
		////////////////////////////////////////////////////
		else if (bps == 16 && !floating_point)
		{
			unsigned char *data = (unsigned char*) malloc((2*nstack)*sizeof(*data));
			tiffile.read(reinterpret_cast<char*>(data), (2*nstack));

			if(!is_bigendian)
			{
				#pragma omp parallel for
				for (long long int pos = 0; pos < nstack; pos++)
				{
					unsigned char temp[4];
					temp[0] = data[2*pos];
					temp[1] = data[2*pos+1];
					output[pos] = reinterpret_cast<int&>(temp);
				}
			}
			else
			{
				#pragma omp parallel for
				for (long long int pos = 0; pos < nstack; pos++)
				{
					unsigned char temp[4];
					temp[1] = data[2*pos];
					temp[0] = data[2*pos+1];
					output[pos] = reinterpret_cast<int&>(temp);
				}
			}
			free(data);
		}
		////////////////////////////////////////////////////

		else
		{
			if(floating_point) std::cout << "Warning! " << bps << " bps and floating point are not handled in Custom3DTifReader" << std::endl;
			else std::cout << "Warning! " << bps << " bps and integer type are not handled in Custom3DTifReader" << std::endl;
		}
		tiffile.close();

		last_bps = bps;

		return output;
	}

    /********************** Reading a single greyscale image as 1D vector **********************/
    std::vector<float> HdCommunication::Get2DTifImage_32bit(std::string file, int outshape[2])
    {
        char *path;
        path = new char[file.length()+1];
        strcpy(path, file.c_str());

        TIFFSetWarningHandler(DummyHandler); //switch of unknown tag warning
        TIFF* tif = TIFFOpen(path, "r");
        vector<float> image;

        if (tif)
        {
            int height, width;
            short s,nsamples,bps;
            tdata_t buf;
            float* data32;
            uint16_t* data16;
            unsigned char tmpval1, tmpval2;

            s = 1; //Setting channels to 1;
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); //dimension1
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); //dimension0
            TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
            TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
            last_bps = bps;

            if ((bps != 32) && (bps != 16) && (bps != 8))
            {
                cout << "Warning! Unidentified bps! Assuming 8bit!" << endl;
                bps = 8;
            }
            if (nsamples > 1)
                cout << "Warning! Potential multichannel image detected!" << endl;

            buf = _TIFFmalloc(TIFFScanlineSize(tif));

            outshape[0] = width;
            outshape[1] = height;

            image.reserve(height*width);

            for (int h = 0; h < height; h++)
            {
                TIFFReadScanline(tif, buf, h, s);

                if (bps == 32)
                {
                    data32 = (float*)buf;
                    for (int w = 0; w < width; w++)
                    {
                        image.push_back(data32[w]);
                    }
                }
                else if (bps ==16)
                {
                    //The image you are trying to read is not 32bit
                    //-> will force conversion assuming this is zero based 16bit tif
                    data16 = (uint16_t*)buf;
                    for (int w = 0; w < width; w++)
                    {
                        image.push_back(data16[w]);
                    }
                }
                else if (bps == 8)
                {
                    //8bit-pointer does not work.
                    //Work around by extracting values from the 16bit-pointer
                    data16 = (uint16_t*)buf;
                    for (int w = 0; w < width/2; w++)
                    {
                        tmpval1 = (unsigned char) data16[w];
                        tmpval2 = (unsigned char) (data16[w]/256);
                        image.push_back(tmpval1);
                        image.push_back(tmpval2);
                    }
                    if (width%2 != 0)
                    {
                        tmpval1 = (unsigned char) data16[width/2];
                        image.push_back(tmpval1);
                    }
                }
            }

            if (bps == 32)
            {
                _TIFFfree(data32);
            }
            else if ((bps == 16) || (bps == 8))
            {
                _TIFFfree(data16);
            }
        }
        else
        {
            cout << "Error! Missing:" << path << endl;
        }
        TIFFClose(tif);
        //TIFFCleanup(tif);
        delete [] path;

        return image;
    }
    float* HdCommunication::Get2DTifImage_32bitPointer(std::string file, int outshape[2])
    {
        char *path;
        path = new char[file.length()+1];
        strcpy(path, file.c_str());

        TIFFSetWarningHandler(DummyHandler); //switch of unknown tag warning
        TIFF* tif = TIFFOpen(path, "r");

        float* image;

        if (tif)
        {
            int height, width;
            short s,nsamples,bps;
            tdata_t buf;
            float* data32;
            uint16_t* data16;
            unsigned char tmpval1, tmpval2;

            s = 1; //Setting channels to 1;
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); //dimension1
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); //dimension0
            TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
            TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
            last_bps = bps;

            if ((bps != 32) && (bps != 16) && (bps != 8))
            {
                cout << "Warning! Unidentified bps! Assuming 8bit!" << endl;
                bps = 8;
            }
            if (nsamples > 1)
                cout << "Warning! Potential multichannel image detected!" << endl;

            buf = _TIFFmalloc(TIFFScanlineSize(tif));

            outshape[0] = width;
            outshape[1] = height;

            image = (float*) malloc(height*width*sizeof(*image));
            int64_t pos = 0;

            for (int h = 0; h < height; h++)
            {
                TIFFReadScanline(tif, buf, h, s);

                if (bps == 32)
                {
                    data32 = (float*)buf;
                    for (int w = 0; w < width; w++)
                    {
                        image[pos] = data32[w];
                        pos++;
                    }
                }
                else if (bps ==16)
                {
                    //The image you are trying to read is not 32bit
                    //-> will force conversion assuming this is zero based 16bit tif
                    data16 = (uint16_t*)buf;
                    for (int w = 0; w < width; w++)
                    {
                        image[pos] = data16[w];
                        pos++;
                    }
                }
                else if (bps == 8)
                {
                    //8bit-pointer does not work.
                    //Work around by extracting values from the 16bit-pointer
                    data16 = (uint16_t*)buf;
                    for (int w = 0; w < width/2; w++)
                    {
                        tmpval1 = (unsigned char) data16[w];
                        tmpval2 = (unsigned char) (data16[w]/256);
                        image[pos] = tmpval1;
                        pos++;
                        image[pos] = tmpval2;
                        pos++;
                    }
                    if (width%2 != 0)
                    {
                        tmpval1 = (unsigned char) data16[width/2];
                        image[pos] = tmpval1;
                        pos++;
                    }
                }
            }

            if (bps == 32)
            {
                _TIFFfree(data32);
            }
            else if ((bps == 16) || (bps == 8))
            {
                _TIFFfree(data16);
            }
        }
        else
        {
            cout << "Error! Missing:" << path << endl;
        }
        TIFFClose(tif);
        //TIFFCleanup(tif);
        delete [] path;
        return image;
    }
    void HdCommunication::Insert2DTifImage_32bitPointer(std::string file, int outshape[2], float *imgpointer, long long int pos0)
	{
		char *path;
		path = new char[file.length()+1];
		strcpy(path, file.c_str());

		TIFFSetWarningHandler(DummyHandler); //switch of unknown tag warning
		TIFF* tif = TIFFOpen(path, "r");

		if (tif)
		{
			int height, width;
			short s,nsamples,bps;
			tdata_t buf;
			float* data32;
			uint16_t* data16;
			unsigned char tmpval1, tmpval2;

			s = 1; //Setting channels to 1;
			TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); //dimension1
			TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); //dimension0
			TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
			TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
			last_bps = bps;

			if ((bps != 32) && (bps != 16) && (bps != 8))
			{
				cout << "Warning! Unidentified bps! Assuming 8bit!" << endl;
				bps = 8;
			}
			buf = _TIFFmalloc(TIFFScanlineSize(tif));

			outshape[0] = width;
			outshape[1] = height;

			int64_t pos = 0;

			for (int h = 0; h < height; h++)
			{
				TIFFReadScanline(tif, buf, h, s);

				if (bps == 32)
				{
					data32 = (float*)buf;
					for (int w = 0; w < width; w++)
					{
						imgpointer[pos+pos0] = data32[w];
						pos++;
					}
				}
				else if (bps ==16)
				{
					//The image you are trying to read is not 32bit
					//-> will force conversion assuming this is zero based 16bit tif
					data16 = (uint16_t*)buf;
					for (int w = 0; w < width; w++)
					{
						imgpointer[pos+pos0] = data16[w];
						pos++;
					}
				}
				else if (bps == 8)
				{
					//8bit-pointer does not work.
					//Work around by extracting values from the 16bit-pointer
					data16 = (uint16_t*)buf;
					for (int w = 0; w < width/2; w++)
					{
						tmpval1 = (unsigned char) data16[w];
						tmpval2 = (unsigned char) (data16[w]/256);
						imgpointer[pos+pos0] = tmpval1;
						pos++;
						imgpointer[pos+pos0] = tmpval2;
						pos++;
					}
					if (width%2 != 0)
					{
						tmpval1 = (unsigned char) data16[width/2];
						imgpointer[pos+pos0] = tmpval1;
						pos++;
					}
				}
			}

			if (bps == 32)
			{
				_TIFFfree(data32);
			}
			else if ((bps == 16) || (bps == 8))
			{
				_TIFFfree(data16);
			}
		}
		else
		{
			cout << "Error! Missing:" << path << endl;
		}
		TIFFClose(tif);

		delete [] path;
		return;
	}
    std::vector<uint8_t> HdCommunication::Get2DTifImage_8bit(std::string file, int outshape[2])
    {
        char *path;
        path = new char[file.length()+1];
        strcpy(path, file.c_str());

        TIFFSetWarningHandler(DummyHandler); //switch of unknown tag warning
        TIFF* tif = TIFFOpen(path, "r");
        vector<uint8_t> image;

        if (tif)
        {
            int height, width;
            short s,nsamples,bps;
            tdata_t buf;
            float* data32;
            uint16_t* data16;
            unsigned char tmpval1, tmpval2;

            s = 1; //Setting channels to 1;
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); //dimension1
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); //dimension0
            TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
            TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bps);
            last_bps = 8;

            if ((bps != 32) && (bps != 16) && (bps != 8))
            {
                cout << "Warning! Unidentified bps! Assuming 8bit!" << endl;
                bps = 8;
            }
            if (nsamples > 1)
                cout << "Warning! Potential multichannel image detected!" << endl;

            buf = _TIFFmalloc(TIFFScanlineSize(tif));

            outshape[0] = width;
            outshape[1] = height;

            image.reserve(height*width);

            for (int h = 0; h < height; h++)
            {
                TIFFReadScanline(tif, buf, h, s);

                if (bps == 32)
                {
                    data32 = (float*)buf;
                    for (int w = 0; w < width; w++)
                        image.push_back(data32[w]);
                }
                else if (bps ==16)
                {
                    //The image you are trying to read is not 32bit
                    //-> will force conversion assuming this is zero based 16bit tif
                    data16 = (uint16_t*)buf;
                    for (int w = 0; w < width; w++)
                        image.push_back(data16[w]);
                }
                else if (bps == 8)
                {
                    //8bit-pointer does not work.
                    //Work around by extracting values from the 16bit-pointer
                    data16 = (uint16_t*)buf;
                    for (int w = 0; w < width/2; w++)
                    {
                        tmpval1 = (unsigned char) data16[w];
                        tmpval2 = (unsigned char) (data16[w]/256);
                        image.push_back(tmpval1);
                        image.push_back(tmpval2);
                    }
                    if (width%2 != 0)
                    {
                        tmpval1 = (unsigned char) data16[width/2];
                        image.push_back(tmpval1);
                    }
                }
            }

            if (bps == 32)
            {
                _TIFFfree(data32);
            }
            else if ((bps == 16) || (bps == 8))
            {
                _TIFFfree(data16);
            }
        }
        else
        {
            cout << "Error! Missing:" << path << endl;
        }
        TIFFClose(tif);
        //TIFFCleanup(tif);
        delete [] path;

        return image;
    }

    /**************** Reading a greyscale image sequence (with overloading) ****************/
    float* HdCommunication::Get3DTifSequence_32bitPointer(std::vector<std::string> &filelist, int outshape[3], bool dspprogress)
	{
    	float* imgstack;

    	char *path;
		path = new char[filelist[0].length()+1];
		strcpy(path, filelist[0].c_str());

    	TIFF* tif = TIFFOpen(path, "r");

    	//check image size
		if (tif)
		{
			 TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &outshape[1]); //dimension1
			 TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &outshape[0]); //dimension0
		     outshape[2] = filelist.size();
		}
		else
		{
			std::cout << "Error! not a tif-sequence!" << std::endl;
			return imgstack;
		}
		TIFFClose(tif);

		//allocate memory
		long long int nslice = outshape[0]*outshape[1];
		long long int nstack = nslice*outshape[2];
		imgstack = (float*) malloc(nstack*sizeof(*imgstack));

		#pragma omp parallel for
		for (unsigned int i=0; i<filelist.size(); i++)
		{
			if (dspprogress)
			{
				printf("Reading slice %u/%lu\r", i+1, filelist.size());
			}

			long long int pos0 = i*nslice;
			Insert2DTifImage_32bitPointer(filelist[i], outshape, imgstack, pos0);
		}

		if (dspprogress)
			printf("Finished reading stack of %u x %u x %u voxels\n",outshape[0],outshape[1],outshape[2]);

		return imgstack;
	}
    std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3], bool dspprogress)
    {
        outshape[2] = filelist.size();
        vector<float> imgstack;

        for (unsigned int i=0; i<filelist.size(); i++)
        {
            if (dspprogress)
            {
                printf("Reading slice %u/%lu\r", i+1, filelist.size());
            }
            vector<float> image = Get2DTifImage_32bit(filelist[i], outshape);
            if (i==0)
            {
            	int64_t nslice = outshape[0]*outshape[1];
                imgstack.reserve(nslice*outshape[2]);
                imgstack.swap(image);
            }
            else
            {
                imgstack.insert(imgstack.end(), image.begin(), image.end());
            }
        }

        if (dspprogress)
            printf("Finished reading stack of %u x %u x %u voxels\n",outshape[0],outshape[1],outshape[2]);
        return imgstack;
    }
    std::vector<uint8_t> HdCommunication::Get3DTifSequence_8bit(std::vector<std::string> &filelist, int outshape[3])
    {
        outshape[2] = filelist.size();
        vector<uint8_t> imgstack;

        for (unsigned int i=0; i<filelist.size(); i++)
        {
            vector<uint8_t> image = Get2DTifImage_8bit(filelist[i], outshape);
            if (i==0)
            {
            	int64_t nslice = outshape[0]*outshape[1];
                imgstack.reserve(nslice*outshape[2]);
                imgstack.swap(image);
            }
            else
            {
                imgstack.insert(imgstack.end(), image.begin(), image.end());
            }
        }
        return imgstack;
    }
    std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::string path, int outshape[3])
    {
        vector<string> filelist;
        GetFilelist(path, filelist);
        vector<float> imgstack = Get3DTifSequence_32bit(filelist, outshape);
        return imgstack;
    }
    std::vector<float> HdCommunication::Get3DTifSequence_32bit(std::vector<std::string> &filelist, int outshape[3])
    {
        vector<float> imgstack;
        imgstack = Get3DTifSequence_32bit(filelist, outshape, false);
        return imgstack;
    }
    std::vector<uint8_t> HdCommunication::Get3DTifSequence_8bit(std::string path, int outshape[3])
    {
        vector<string> filelist;
        GetFilelist(path, filelist);
        vector<uint8_t> imgstack = Get3DTifSequence_8bit(filelist, outshape);
        return imgstack;
    }

    /********************** Saving a single greyscale image **********************/
    void HdCommunication::Save2DTifImage_8bit(uint8_t *image, int imgshape[2], std::string path, std::string name)
    {
        int64_t pos = 0;

        TIFF *output_image;
        string filename = path + "/" + name + ".tif";
        int width = imgshape[0];
        int height = imgshape[1];

        //Create directory if necessary
        //********************************************
        char *dir_path;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        //********************************************

        // Open the TIFF file
        //********************************************
        if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
            cerr << "Unable to write tif file: " << filename << endl;
        //********************************************

        //set rows per strip
        //according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
        long rowsperstrip = (long) 2048/width;

        if (rowsperstrip == 0) rowsperstrip = 1;

        // Set basic tags
        //********************************************
        TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

        TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        //********************************************

        // Write the information to the file
        //********************************************
        tdata_t buf;

        buf = _TIFFmalloc(TIFFScanlineSize(output_image));
        for (int h = 0; h < height; h++)
        {
            buf = &image[pos];
            pos += width;
            TIFFWriteScanline(output_image, buf, h, 1);
        }
        //********************************************

        // Close the file
        TIFFClose(output_image);
        return;
    }
    void HdCommunication::Save2DTifImage_8bit(std::vector<uint8_t> &image, int imgshape[2], std::string path, std::string name, int64_t pos)
    {
        TIFF *output_image;
        string filename = path + "/" + name + ".tif";
        int width = imgshape[0];
        int height = imgshape[1];

        //Create directory if necessary
        //********************************************
        char *dir_path;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        //********************************************

        // Open the TIFF file
        //********************************************
        if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
            cerr << "Unable to write tif file: " << filename << endl;
        //********************************************

        //set rows per strip
        //according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
        long rowsperstrip = (long) 2048/width;

        if (rowsperstrip == 0) rowsperstrip = 1;

        // Set basic tags
        //********************************************
        TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

        TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        //********************************************

        // Write the information to the file
        //********************************************
        vector<uint8_t> scanline;
        scanline.reserve(width);
        tdata_t buf;

        buf = _TIFFmalloc(TIFFScanlineSize(output_image));
        for (int h = 0; h < height; h++)
        {
            buf = &image[pos];
            pos += width;
            TIFFWriteScanline(output_image, buf, h, 1);
        }
        //********************************************

        // Close the file
        TIFFClose(output_image);
        return;
    }

    void HdCommunication::Save2DTifImage_32bit(int *image, int imgshape[2], std::string path, std::string name)
    {
        int64_t pos = 0;

        TIFF *output_image;
        string filename = path + "/" + name + ".tif";
        int width = imgshape[0];
        int height = imgshape[1];

        //Create directory if necessary
        //********************************************
        char *dir_path;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        //********************************************

        // Open the TIFF file
        //********************************************
        if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
            cerr << "Unable to write tif file: " << filename << endl;
        //********************************************

        //set rows per strip
        //according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
        long rowsperstrip = (long) 2048/width;

        if (rowsperstrip == 0) rowsperstrip = 1;

        // Set basic tags
        //********************************************
        TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

        TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
        //********************************************

        // Write the information to the file
        //********************************************
        vector<int> scanline;
        scanline.reserve(width);
        tdata_t buf;

        buf = _TIFFmalloc(TIFFScanlineSize(output_image));
        for (int h = 0; h < height; h++)
        {
            buf = &image[pos];
            pos += width;
            TIFFWriteScanline(output_image, buf, h, 1);
        }
        //********************************************

        // Close the file
        TIFFClose(output_image);
        return;
    }
    void HdCommunication::Save2DTifImage_32bit(int *image, int imgshape[2], std::string path, std::string name, int64_t pos)
    {
        TIFF *output_image;
        string filename = path + "/" + name + ".tif";
        int width = imgshape[0];
        int height = imgshape[1];

        //Create directory if necessary
        //********************************************
        char *dir_path;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        //********************************************

        // Open the TIFF file
        //********************************************
        if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
            cerr << "Unable to write tif file: " << filename << endl;
        //********************************************

        //set rows per strip
        //according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
        long rowsperstrip = (long) 2048/width;

        if (rowsperstrip == 0) rowsperstrip = 1;

        // Set basic tags
        //********************************************

        TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

        TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_INT);
        //********************************************

        // Write the information to the file
        //********************************************
        vector<float> scanline;
        scanline.reserve(width);
        tdata_t buf;

        buf = _TIFFmalloc(TIFFScanlineSize(output_image));
        for (int h = 0; h < height; h++)
        {
            buf = &image[pos];
            pos += width;
            TIFFWriteScanline(output_image, buf, h, 1);
        }
        //********************************************

        // Close the file
        TIFFClose(output_image);
        return;
    }
    void HdCommunication::Save2DTifImage_32bit(std::vector<float> &image, int imgshape[2], std::string path, std::string name, int64_t pos)
    {
        TIFF *output_image;
        string filename = path + "/" + name + ".tif";
        int width = imgshape[0];
        int height = imgshape[1];

        //Create directory if necessary
        //********************************************
        char *dir_path;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        //********************************************

        // Open the TIFF file
        //********************************************
        if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
            cerr << "Unable to write tif file: " << filename << endl;
        //********************************************

        //set rows per strip
        //according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
        long rowsperstrip = (long) 2048/width;

        if (rowsperstrip == 0) rowsperstrip = 1;

        // Set basic tags
        //********************************************

        TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 32);
        TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

        TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
        TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
        //********************************************

        // Write the information to the file
        //********************************************
        vector<float> scanline;
        scanline.reserve(width);
        tdata_t buf;

        buf = _TIFFmalloc(TIFFScanlineSize(output_image));
        for (int h = 0; h < height; h++)
        {
            buf = &image[pos];
            pos += width;
            TIFFWriteScanline(output_image, buf, h, 1);
        }
        //********************************************

        // Close the file
        TIFFClose(output_image);
        return;
    }
    void HdCommunication::Save2DTifImage_32bit(float *image, int imgshape[2], std::string path, std::string name, int64_t pos)
	{
		TIFF *output_image;
		string filename = path + "/" + name + ".tif";
		int width = imgshape[0];
		int height = imgshape[1];

		//Create directory if necessary
		//********************************************
		char *dir_path;
		dir_path = new char[path.length()+1];
		strcpy(dir_path, path.c_str());

		if (!path_exists(dir_path)) makepath(dir_path);
		//********************************************

		// Open the TIFF file
		//********************************************
		if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
			cerr << "Unable to write tif file: " << filename << endl;
		//********************************************

		//set rows per strip
		//according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
		long rowsperstrip = (long) 2048/width;

		if (rowsperstrip == 0) rowsperstrip = 1;

		// Set basic tags
		//********************************************

		TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
		TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
		TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 32);
		TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
		TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

		TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
		TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
		//********************************************

		// Write the information to the file
		//********************************************
		vector<float> scanline;
		scanline.reserve(width);
		tdata_t buf;

		buf = _TIFFmalloc(TIFFScanlineSize(output_image));
		for (int h = 0; h < height; h++)
		{
			buf = &image[pos];
			pos += width;
			TIFFWriteScanline(output_image, buf, h, 1);
		}
		//********************************************

		// Close the file
		TIFFClose(output_image);
		return;
	}
    void HdCommunication::Save2DTifImage_RGB(uint8_t *image, int imgshape[2], std::string path, std::string name)
	{
		int64_t pos = 0;

		TIFF *output_image;
		string filename = path + "/" + name + ".tif";
		int width = imgshape[0];
		int height = imgshape[1];

		//Create directory if necessary
		//********************************************
		char *dir_path;
		dir_path = new char[path.length()+1];
		strcpy(dir_path, path.c_str());

		if (!path_exists(dir_path)) makepath(dir_path);
		//********************************************

		// Open the TIFF file
		//********************************************
		if((output_image = TIFFOpen(filename.c_str(), "w")) == NULL)
			cerr << "Unable to write tif file: " << filename << endl;
		//********************************************

		//set rows per strip
		//according to libtiff-docs 8k-byte are optimal, i.e. 2048 pixels in a 32bit-image
		long rowsperstrip = (long) 2048/width;

		if (rowsperstrip == 0) rowsperstrip = 1;

		// Set basic tags
		//********************************************
		TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, width);
		TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, height);
		TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
		TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 3);
		TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
		TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP, rowsperstrip);

		TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
		TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
		TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
		//********************************************

		// Write the information to the file
		//********************************************
		tdata_t buf;

		buf = _TIFFmalloc(TIFFScanlineSize(output_image));
		for (int h = 0; h < height; h++)
		{
			buf = &image[pos];
			pos += 3*width;
			TIFFWriteScanline(output_image, buf, h, 1);
		}
		//********************************************

		// Close the file
		TIFFClose(output_image);
		return;
	}

    /********************** Saving a greyscale sequence **********************/
    void HdCommunication::SaveTifSequence_32bit(std::vector<float> &image, int imgshape[3], std::string path, std::string name, bool dspprogress)
    {
        const int zfill = 4;
        int64_t nslice = imgshape[0]*imgshape[1];

        for(int i=0; i<imgshape[2]; i++)
        {
            if (dspprogress)
                printf("Saving %s %u/%u\r", name.c_str(), i+1, imgshape[2]);

            //Create filename
            string id = to_string(i);
            while(id.length()<zfill)
                id = "0"+id;

            //Call 2D-Save at position in vector that is the beginning of the current slice
            Save2DTifImage_32bit(image, imgshape, path, name+id, i*nslice);
        }

        if (dspprogress)
            printf("\n");

        return;
    }
    void HdCommunication::SaveTifSequence_32bit(float *image, int imgshape[3], std::string path, std::string name, bool dspprogress)
	{
		const int zfill = 4;
		int64_t nslice = imgshape[0]*imgshape[1];

		for(int i=0; i<imgshape[2]; i++)
		{
			if (dspprogress)
				printf("Saving %s %u/%u\r", name.c_str(), i+1, imgshape[2]);

			//Create filename
			string id = to_string(i);
			while(id.length()<zfill)
				id = "0"+id;

			//Call 2D-Save at position in vector that is the beginning of the current slice

			Save2DTifImage_32bit(image, imgshape, path, name+id, i*nslice);
		}

		if (dspprogress)
			printf("\n");

		return;
	}
    void HdCommunication::SaveTifSequence_32bit(int *image, int imgshape[3], std::string path, std::string name)
    {
        const int zfill = 4;
        int64_t nslice = imgshape[0]*imgshape[1];

        for(int i=0; i<imgshape[2]; i++)
        {
            //Create filename
            string id = to_string(i);
            while(id.length()<zfill)
                id = "0"+id;

            long long int pos = i*nslice;

            //Call 2D-Save at position in vector that is the beginning of the current slice
            Save2DTifImage_32bit(image, imgshape, path, name+id, pos);
        }
        return;
    }
    void HdCommunication::SaveTifSequence_8bit(std::vector<uint8_t> &image, int imgshape[3], std::string path, std::string name, int firstslice)
    {
        const int zfill = 4;
        int64_t nslice = imgshape[0]*imgshape[1];


        for(int i=0; i<imgshape[2]; i++)
        {
            //Create filename
            string id = to_string(i+firstslice);
            while(id.length()<zfill)
                id = "0"+id;

            //Call 2D-Save at position in vector that is the beginning of the current slice
            Save2DTifImage_8bit(image, imgshape, path, name+id, i*nslice);
        }
    }

    /********************** Saving a vector **********************/
    void HdCommunication::Save3DVector_vtk(float *u, int shape[3], std::string path, std::string name, std::string header)
    {
        std::FILE *output_image;
        string filename = path + "/" + name + ".vtk";
        int width = shape[0];
        int height = shape[1];
        int depth = shape[2];
        long long int nslice = shape[0]*shape[1];
        long long int nstack = nslice*shape[2];

        //Create directory if necessary
        //********************************************
        char *dir_path, *header_;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        header += "\n";
        header_ = new char[header.length()+1];
        strcpy(header_, header.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        //********************************************

        // Open the VTK file
        //********************************************
        if((output_image = std::fopen(filename.c_str(), "wb")) == NULL)
            cerr << "Unable to write vtk file: " << filename << endl;
        //********************************************

        if (output_image && shape[2] > 1)
        {
            std::fprintf(output_image, "# vtk DataFile Version 2.0\n");
            std::fprintf(output_image, "%s", header_);
            std::fprintf(output_image, "ASCII\n");
            std::fprintf(output_image, "DATASET STRUCTURED_POINTS\n");
            std::fprintf(output_image, "DIMENSIONS %d %d %d\n", width, height, depth);
            std::fprintf(output_image, "ORIGIN 0 0 0\n");
            std::fprintf(output_image, "SPACING 1 1 1\n");
            std::fprintf(output_image, "POINT_DATA %d\n", width*height*depth);
            std::fprintf(output_image, "VECTORS vectors float\n");

            for (long long int pos = 0; pos < nstack; pos++)
            {
                 std::fprintf(output_image, "%f %f %f ", u[pos], u[pos+nstack], u[pos+2*nstack]);
            }
            std::fclose(output_image);
          }
          else
          {
            std::fprintf(output_image, "# vtk DataFile Version 2.0\n");
            std::fprintf(output_image, "%s", header_);
            std::fprintf(output_image, "ASCII\n");
            std::fprintf(output_image, "DATASET STRUCTURED_POINTS\n");
            std::fprintf(output_image, "DIMENSIONS %d %d %d\n", width, height, 1);
            std::fprintf(output_image, "ORIGIN 0 0 0\n");
            std::fprintf(output_image, "SPACING 1 1 1\n");
            std::fprintf(output_image, "POINT_DATA %d\n", width*height);
            std::fprintf(output_image, "VECTORS vectors float\n");

            for (long long int pos = 0; pos < nslice; pos++)
            {
                 std::fprintf(output_image, "%f %f %f ", u[pos], u[pos+nstack], 0.0f);
            }
            std::fclose(output_image);
          }

          return;
    }

    /********************** Helper functions **********************/
    int HdCommunication::GetFilelist(std::string const dir, std::vector<std::string> &files)
    {
        int n_files = 0;
        DIR *dp;
        struct dirent *dirp;
        if((dp  = opendir(dir.c_str())) == NULL)
        {
            cout << "Error(" << errno << ") opening " << dir << endl;
            return errno;
        }

        while ((dirp = readdir(dp)) != NULL)
        {
            if ((hasEnding(dirp->d_name,".tif")) || (hasEnding(dirp->d_name,".tiff")))
            {
                files.push_back(dir+string(dirp->d_name));
                n_files++;
            }
        }
        closedir(dp);
        std::sort(files.begin(),files.end());
        return n_files;
    }
    std::vector<std::string> HdCommunication::GetFilelist(std::string datapath, int outshape[3])
    {
        std::vector<std::string> filelist;
        int depth = GetFilelist(datapath, filelist);

        char *path;
        path = new char[filelist[0].length()+1];
        strcpy(path, filelist[0].c_str());

        TIFFSetWarningHandler(DummyHandler); //switch of unknown tag warning
        TIFF* tif = TIFFOpen(path, "r");

        int height, width;
        if (tif)
        {
            TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &height); //dimension1
            TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &width); //dimension0
        }

        outshape[0] = width;
        outshape[1] = height;
        outshape[2] = depth;
        return filelist;
    }
    bool HdCommunication::hasEnding (std::string const &fullString, std::string const &ending)
    {
        if (fullString.length() >= ending.length())
        {
            return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
        }
        else
        {
            return false;
        }
    }
    void HdCommunication::makedir(const std::string path)
    {
        char *dir_path;
        dir_path = new char[path.length()+1];
        strcpy(dir_path, path.c_str());

        if (!path_exists(dir_path)) makepath(dir_path);
        return;
    }
}
