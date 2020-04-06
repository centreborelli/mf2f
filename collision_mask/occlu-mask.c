#include "argparse.h"   // command line parser
#include "iio.h"        // image i/o

// some macros

#define max(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a > _b ? _a : _b; })

#define min(a,b) \
	({ __typeof__ (a) _a = (a); \
	   __typeof__ (b) _b = (b); \
	   _a < _b ? _a : _b; })

// 'usage' message in the command line
static const char *const usages[] = {
	"occlu-mask -i flow-path -o occl-mask-path [optional args]",
	NULL,
};

// main
int main(int argc, const char *argv[])
{
//	omp_set_num_threads(2);
	// parse command line [[[2

	// command line parameters and their defaults
	const char *flow_path = NULL; // input flow path
	const char *dens_path = NULL; // output density map path
	const char *occl_path = NULL; // output occlusion path
	bool verbose = false;
	int  verbose_int = 0; // hack around bug in argparse

	float thres = 0; // -1 means automatic value

	// configure command line parser
	struct argparse_option options[] = {
		OPT_HELP(),
		OPT_GROUP("Data i/o options"),
		OPT_STRING ('i', "flow" , &flow_path, "input optical flow path"),
		OPT_STRING ('o', "occl" , &occl_path, "output occlusion mask path"),
		OPT_STRING ('d', "dens" , &dens_path, "output density map path"),
		OPT_GROUP("Parameters"),
		OPT_FLOAT  ('t', "threshold", &thres, "threshold parameter"),
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
		return fprintf(stderr, "Error: no input path given, exiting\n"), 1;

	if (!occl_path)
		return fprintf(stderr, "Error: no output path given, exiting\n"), 1;

	// print parameters
	if (verbose)
	{
		printf("data input:\n");
		printf("\tflow     %s\n", flow_path);
		printf("\tocclu.   %s\n", occl_path);
		printf("\n");

		printf("parameters:\n");
		printf("\tthres      %g\n", thres);
		printf("\n");
	}

	// load data [[[2
	int w, h, c;
	float *flow = iio_read_image_float_vec(flow_path, &w, &h, &c);
	if (!flow)
		return fprintf(stderr, "Error while openning bwd optical flow\n"), 1;

	if (c != 2) 
	{
		if (flow) free(flow);
		return fprintf(stderr, "Optical flow should have 2 channels\n"), 1;
	}

	// compute arrival density [[[2
	const int wh = w*h;
	float * dens = malloc(wh*sizeof(float)); // arrival density
	float (*D)[w]    = (void *)dens;
	float (*F)[w][c] = (void *)flow;
	for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
		float vx = x + F[y][x][0];
		float vy = y + F[y][x][1];
		int vx0 = vx, vy0 = vy;

		if (vx0 >= 0 && vy0 >= 0 && vx0+1 < w && vy0+1 < h)
		{
			// compute bilinear weights
			float vxx = vx - (float)vx0;
			float vyy = vy - (float)vy0;

			// aggregate on occlusion mask
			D[vy0+0][vx0+0] += (1 - vxx)*(1 - vyy);
			D[vy0+0][vx0+1] +=      vxx *(1 - vyy);
			D[vy0+1][vx0+0] += (1 - vxx)*     vyy ;
			D[vy0+1][vx0+1] +=      vxx *     vyy ;
		}
	}

	// compute occlusions [[[2
	float * occl = malloc(wh*sizeof(float)); // occlusion mask
	float (*K)[w] = (void *)occl;
	for (int y = 0; y < h; ++y)
	for (int x = 0; x < w; ++x)
	{
		float vx = x + F[y][x][0];
		float vy = y + F[y][x][1];
		int vx0 = vx, vy0 = vy;

		if (vx0 >= 0 && vy0 >= 0 && vx0+1 < w && vy0+1 < h)
		{
			// evaluate density at arrival point

			// 'max-pulling' interpolation
			float d = max(max(D[vy0+0][vx0+0], D[vy0+0][vx0+1]),
			              max(D[vy0+1][vx0+0], D[vy0+1][vx0+1]));

//			// bilinear interpolation
//			float vxx = vx - (float)vx0;
//			float vyy = vy - (float)vy0;
//			float d = D[vy0+0][vx0+0] * (1 - vxx)*(1 - vyy) +
//			          D[vy0+0][vx0+1] *      vxx *(1 - vyy) +
//			          D[vy0+1][vx0+0] * (1 - vxx)*     vyy  +
//			          D[vy0+1][vx0+1] *      vxx *     vyy  ;

			// occlusions are defined as high density areas
			K[y][x] = (thres) ? d > thres : d;
		}
		else
			K[y][x] = (thres) ? 2. : -1.;
	}

	// write result and quit [[[2
	iio_write_image_float_vec(occl_path, occl, w, h, 1);
	if (dens_path)
		iio_write_image_float_vec(dens_path, dens, w, h, 1);

	if (flow) free(flow);
	if (dens) free(dens);
	if (occl) free(occl);

	return EXIT_SUCCESS; // ]]]2
}

// vim:set foldmethod=marker:
// vim:set foldmarker=[[[,]]]:
