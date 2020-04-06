#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

#include <stdlib.h>
#include <math.h>       // nans (used as boundary value by bicubic interp)
#include <fftw3.h>      // computes dct
#include <omp.h>

#include <stdio.h>     // getchar() for debugging

// some macros [[[1

#define max(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a > _b ? _a : _b; })

#define min(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a < _b ? _a : _b; })

// bicubic interpolation [[[1

#ifdef NAN
// extrapolate by nan
float getsample_nan(float *x, int w, int h, int pd, int i, int j, int l)
{
	assert(l >= 0 && l < pd);
	return (i < 0 || i >= w || j < 0 || j >= h) ? NAN : x[(i + j*w)*pd + l];
}
#endif//NAN

float cubic_interpolation(float v[4], float x)
{
	return v[1] + 0.5 * x*(v[2] - v[0]
			+ x*(2.0*v[0] - 5.0*v[1] + 4.0*v[2] - v[3]
			+ x*(3.0*(v[1] - v[2]) + v[3] - v[0])));
}

float bicubic_interpolation_cell(float p[4][4], float x, float y)
{
	float v[4];
	v[0] = cubic_interpolation(p[0], y);
	v[1] = cubic_interpolation(p[1], y);
	v[2] = cubic_interpolation(p[2], y);
	v[3] = cubic_interpolation(p[3], y);
	return cubic_interpolation(v, x);
}

void bicubic_interpolation_nans(float *result,
		float *img, int w, int h, int pd, float x, float y)
{
	x -= 1;
	y -= 1;

	int ix = floor(x);
	int iy = floor(y);
	for (int l = 0; l < pd; l++) {
		float c[4][4];
		for (int j = 0; j < 4; j++)
		for (int i = 0; i < 4; i++)
			c[i][j] = getsample_nan(img, w, h, pd, ix + i, iy + j, l);
		float r = bicubic_interpolation_cell(c, x - ix, y - iy);
		result[l] = r;
	}
}

void warp_bicubic(float *imw, float *im, float *of,
		int w, int h, int ch)
{
	// warp previous frame
	for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
		float xw = x + of[(x + y*w)*2 + 0];
		float yw = y + of[(x + y*w)*2 + 1];
		bicubic_interpolation_nans(imw + (x + y*w)*ch, im, w, h, ch, xw, yw);
	}

	return;
}


// main [[[1

// 'usage' message in the command line
static const char *const usages[] = {
	"backflow++ -i image-path -f flow-path -o warped-image-path [optional args]",
	NULL,
};

int main(int argc, const char *argv[])
{
	// parse command line [[[2

	// command line parameters and their defaults
	const char *flow_path = NULL; // input flow path
	const char *imge_path = NULL; // input image path
	const char *warp_path = NULL; // output warped image path
	bool verbose = false;
	int  verbose_int = 0; // hack around bug in argparse

	// configure command line parser
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Data i/o options"),
		OPT_STRING ('i', "", &imge_path, "path to input image"),
		OPT_STRING ('f', "", &flow_path, "path to input optical flow"),
		OPT_STRING ('o', "", &warp_path, "path to output warped image"),
		OPT_GROUP("Parameters"),
		OPT_INTEGER('v', "verbose", &verbose_int, "verbose output"),
		OPT_END(),
	};

	// parse command line
	struct argparse argparse;
	argparse_init(&argparse, options, usages, 0);
	argparse_describe(&argparse, "\nOcclusion detection as pre-image of high density regions.", "");
	argc = argparse_parse(&argparse, argc, argv);

	// hack around argparse bug
	verbose = (bool)verbose_int;

	// check if i/o paths have been provided
	if (!flow_path)
		return fprintf(stderr, "Error: no input flow path given, exiting\n"), 1;

	if (!imge_path)
		return fprintf(stderr, "Error: no input image path given, exiting\n"), 1;

	if (!warp_path)
		return fprintf(stderr, "Error: no output path given, exiting\n"), 1;

	// print parameters
	if (verbose)
	{
		printf("data input:\n");
		printf("\tflow    %s\n", flow_path);
		printf("\timage   %s\n", imge_path);
		printf("\twarp    %s\n", warp_path);
		printf("\n");
	}

	// load data [[[2
	int w, h, c;
	float *imge = iio_read_image_float_vec(imge_path, &w, &h, &c);
	if (!imge)
		return fprintf(stderr, "Error while openning image\n"), 1;

	int w1, h1, c1;
	float *flow = iio_read_image_float_vec(flow_path, &w1, &h1, &c1);
	if (!flow)
		return fprintf(stderr, "Error while openning optical flow\n"), 1;

	if (c1 != 2) 
	{
		if (flow) free(flow);
		return fprintf(stderr, "Optical flow should have 2 channels\n"), 1;
	}

	if (w != w1 || h != h1) 
	{
		if (flow) free(flow);
		if (imge) free(imge);
		return fprintf(stderr, "Flow and image dimensions don't match\n"), 1;
	}

	// compute bicubic warping
	float * warp = malloc(w*h*c*sizeof(float)); // arrival density
	warp_bicubic(warp, imge, flow, w, h, c);

	// write result and quit
	iio_write_image_float_vec(warp_path, warp, w, h, c);

	if (flow) free(flow);
	if (imge) free(imge);
	if (warp) free(warp);

	return EXIT_SUCCESS; // ]]]2
}

// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
