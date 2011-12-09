#include "opj_convert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Get logarithm of an integer and round downwards.
 *
 * log2(a)
 */
static int int_floorlog2(int a) {
	int l;
	for (l = 0; a > 1; l++) {
		a >>= 1;
	}
	return l;
}

/*
 * Divide an integer by a power of 2 and round upwards.
 *
 * a divided by 2^b
 */
static int int_ceildivpow2(int a, int b) {
	return (a + (1 << b) - 1) >> b;
}

/*
 * Divide an integer and round upwards.
 *
 * a divided by b
 */
static int int_ceildiv(int a, int b) {
	return (a + b - 1) / b;
}
/* WORD defines a two byte word */
typedef unsigned short int WORD;

/* DWORD defines a four byte word */
typedef unsigned long int DWORD;

typedef struct {
  WORD bfType;			/* 'BM' for Bitmap (19776) */
  DWORD bfSize;			/* Size of the file        */
  WORD bfReserved1;		/* Reserved : 0            */
  WORD bfReserved2;		/* Reserved : 0            */
  DWORD bfOffBits;		/* Offset                  */
} BITMAPFILEHEADER_t;

typedef struct {
  DWORD biSize;			/* Size of the structure in bytes */
  DWORD biWidth;		/* Width of the image in pixels */
  DWORD biHeight;		/* Heigth of the image in pixels */
  WORD biPlanes;		/* 1 */
  WORD biBitCount;		/* Number of color bits by pixels */
  DWORD biCompression;		/* Type of encoding 0: none 1: RLE8 2: RLE4 */
  DWORD biSizeImage;		/* Size of the image in bytes */
  DWORD biXpelsPerMeter;	/* Horizontal (X) resolution in pixels/meter */
  DWORD biYpelsPerMeter;	/* Vertical (Y) resolution in pixels/meter */
  DWORD biClrUsed;		/* Number of color used in the image (0: ALL) */
  DWORD biClrImportant;		/* Number of important color (0: ALL) */
} BITMAPINFOHEADER_t;

opj_image_t*
opj_bmp_to_image(const char *filename, opj_cparameters_t *parameters)
{
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;

	int i, numcomps, w, h;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t cmptparm[3];	/* maximum of 3 components */
	opj_image_t * image = NULL;

	FILE *IN;
	BITMAPFILEHEADER_t File_h;
	BITMAPINFOHEADER_t Info_h;
	unsigned char *RGB;
	unsigned char *table_R, *table_G, *table_B;
	unsigned int j, PAD = 0;

	int x, y, index;
	int gray_scale = 1, not_end_file = 1; 

	unsigned int line = 0, col = 0;
	unsigned char v, v2;
	DWORD W, H;
  
	IN = fopen(filename, "rb");
	if (!IN) {
		fprintf(stderr, "Failed to open %s for reading !!\n", filename);
		return 0;
	}
	
	File_h.bfType = getc(IN);
	File_h.bfType = (getc(IN) << 8) + File_h.bfType;
	
	if (File_h.bfType != 19778) {
		fprintf(stderr,"Error, not a BMP file!\n");
		return 0;
	} else {
		/* FILE HEADER */
		/* ------------- */
		File_h.bfSize = getc(IN);
		File_h.bfSize = (getc(IN) << 8) + File_h.bfSize;
		File_h.bfSize = (getc(IN) << 16) + File_h.bfSize;
		File_h.bfSize = (getc(IN) << 24) + File_h.bfSize;

		File_h.bfReserved1 = getc(IN);
		File_h.bfReserved1 = (getc(IN) << 8) + File_h.bfReserved1;

		File_h.bfReserved2 = getc(IN);
		File_h.bfReserved2 = (getc(IN) << 8) + File_h.bfReserved2;

		File_h.bfOffBits = getc(IN);
		File_h.bfOffBits = (getc(IN) << 8) + File_h.bfOffBits;
		File_h.bfOffBits = (getc(IN) << 16) + File_h.bfOffBits;
		File_h.bfOffBits = (getc(IN) << 24) + File_h.bfOffBits;

		/* INFO HEADER */
		/* ------------- */

		Info_h.biSize = getc(IN);
		Info_h.biSize = (getc(IN) << 8) + Info_h.biSize;
		Info_h.biSize = (getc(IN) << 16) + Info_h.biSize;
		Info_h.biSize = (getc(IN) << 24) + Info_h.biSize;

		Info_h.biWidth = getc(IN);
		Info_h.biWidth = (getc(IN) << 8) + Info_h.biWidth;
		Info_h.biWidth = (getc(IN) << 16) + Info_h.biWidth;
		Info_h.biWidth = (getc(IN) << 24) + Info_h.biWidth;
		w = Info_h.biWidth;

		Info_h.biHeight = getc(IN);
		Info_h.biHeight = (getc(IN) << 8) + Info_h.biHeight;
		Info_h.biHeight = (getc(IN) << 16) + Info_h.biHeight;
		Info_h.biHeight = (getc(IN) << 24) + Info_h.biHeight;
		h = Info_h.biHeight;

		Info_h.biPlanes = getc(IN);
		Info_h.biPlanes = (getc(IN) << 8) + Info_h.biPlanes;

		Info_h.biBitCount = getc(IN);
		Info_h.biBitCount = (getc(IN) << 8) + Info_h.biBitCount;

		Info_h.biCompression = getc(IN);
		Info_h.biCompression = (getc(IN) << 8) + Info_h.biCompression;
		Info_h.biCompression = (getc(IN) << 16) + Info_h.biCompression;
		Info_h.biCompression = (getc(IN) << 24) + Info_h.biCompression;

		Info_h.biSizeImage = getc(IN);
		Info_h.biSizeImage = (getc(IN) << 8) + Info_h.biSizeImage;
		Info_h.biSizeImage = (getc(IN) << 16) + Info_h.biSizeImage;
		Info_h.biSizeImage = (getc(IN) << 24) + Info_h.biSizeImage;

		Info_h.biXpelsPerMeter = getc(IN);
		Info_h.biXpelsPerMeter = (getc(IN) << 8) + Info_h.biXpelsPerMeter;
		Info_h.biXpelsPerMeter = (getc(IN) << 16) + Info_h.biXpelsPerMeter;
		Info_h.biXpelsPerMeter = (getc(IN) << 24) + Info_h.biXpelsPerMeter;

		Info_h.biYpelsPerMeter = getc(IN);
		Info_h.biYpelsPerMeter = (getc(IN) << 8) + Info_h.biYpelsPerMeter;
		Info_h.biYpelsPerMeter = (getc(IN) << 16) + Info_h.biYpelsPerMeter;
		Info_h.biYpelsPerMeter = (getc(IN) << 24) + Info_h.biYpelsPerMeter;

		Info_h.biClrUsed = getc(IN);
		Info_h.biClrUsed = (getc(IN) << 8) + Info_h.biClrUsed;
		Info_h.biClrUsed = (getc(IN) << 16) + Info_h.biClrUsed;
		Info_h.biClrUsed = (getc(IN) << 24) + Info_h.biClrUsed;

		Info_h.biClrImportant = getc(IN);
		Info_h.biClrImportant = (getc(IN) << 8) + Info_h.biClrImportant;
		Info_h.biClrImportant = (getc(IN) << 16) + Info_h.biClrImportant;
		Info_h.biClrImportant = (getc(IN) << 24) + Info_h.biClrImportant;

		/* Read the data and store them in the OUT file */
    
		if (Info_h.biBitCount == 24) {
			numcomps = 3;
			color_space = CLRSPC_SRGB;
			/* initialize image components */
			memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
			for(i = 0; i < numcomps; i++) {
				cmptparm[i].prec = 8;
				cmptparm[i].bpp = 8;
				cmptparm[i].sgnd = 0;
				cmptparm[i].dx = subsampling_dx;
				cmptparm[i].dy = subsampling_dy;
				cmptparm[i].w = w;
				cmptparm[i].h = h;
			}
			/* create the image */
			image = opj_image_create(numcomps, &cmptparm[0], color_space);
			if(!image) {
				fclose(IN);
				return NULL;
			}

			/* set image offset and reference grid */
			image->x0 = parameters->image_offset_x0;
			image->y0 = parameters->image_offset_y0;
			image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
			image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

			/* set image data */

			/* Place the cursor at the beginning of the image information */
			fseek(IN, 0, SEEK_SET);
			fseek(IN, File_h.bfOffBits, SEEK_SET);
			
			W = Info_h.biWidth;
			H = Info_h.biHeight;

			/* PAD = 4 - (3 * W) % 4; */
			/* PAD = (PAD == 4) ? 0 : PAD; */
			PAD = (3 * W) % 4 ? 4 - (3 * W) % 4 : 0;
			
			RGB = (unsigned char *) malloc((3 * W + PAD) * H * sizeof(unsigned char));
			
			size_t result = fread(RGB, sizeof(unsigned char), (3 * W + PAD) * H, IN);
			
			index = 0;

			for(y = 0; y < (int)H; y++) {
				unsigned char *scanline = RGB + (3 * W + PAD) * (H - 1 - y);
				for(x = 0; x < (int)W; x++) {
					unsigned char *pixel = &scanline[3 * x];
					image->comps[0].data[index] = pixel[2];	/* R */
					image->comps[1].data[index] = pixel[1];	/* G */
					image->comps[2].data[index] = pixel[0];	/* B */
					index++;
				}
			}

			free(RGB);

		} else if (Info_h.biBitCount == 8 && Info_h.biCompression == 0) {
			table_R = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_G = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_B = (unsigned char *) malloc(256 * sizeof(unsigned char));
			
			for (j = 0; j < Info_h.biClrUsed; j++) {
				table_B[j] = getc(IN);
				table_G[j] = getc(IN);
				table_R[j] = getc(IN);
				getc(IN);
				if (table_R[j] != table_G[j] && table_R[j] != table_B[j] && table_G[j] != table_B[j])
					gray_scale = 0;
			}
			
			/* Place the cursor at the beginning of the image information */
			fseek(IN, 0, SEEK_SET);
			fseek(IN, File_h.bfOffBits, SEEK_SET);
			
			W = Info_h.biWidth;
			H = Info_h.biHeight;
			if (Info_h.biWidth % 2)
				W++;
			
			numcomps = gray_scale ? 1 : 3;
			color_space = gray_scale ? CLRSPC_GRAY : CLRSPC_SRGB;
			/* initialize image components */
			memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
			for(i = 0; i < numcomps; i++) {
				cmptparm[i].prec = 8;
				cmptparm[i].bpp = 8;
				cmptparm[i].sgnd = 0;
				cmptparm[i].dx = subsampling_dx;
				cmptparm[i].dy = subsampling_dy;
				cmptparm[i].w = w;
				cmptparm[i].h = h;
			}
			/* create the image */
			image = opj_image_create(numcomps, &cmptparm[0], color_space);
			if(!image) {
				fclose(IN);
				return NULL;
			}

			/* set image offset and reference grid */
			image->x0 = parameters->image_offset_x0;
			image->y0 = parameters->image_offset_y0;
			image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
			image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

			/* set image data */

			RGB = (unsigned char *) malloc(W * H * sizeof(unsigned char));
			
			size_t result = fread(RGB, sizeof(unsigned char), W * H, IN);
			if (gray_scale) {
				index = 0;
				for (j = 0; j < W * H; j++) {
					if ((j % W < W - 1 && Info_h.biWidth % 2) || !(Info_h.biWidth % 2)) {
						image->comps[0].data[index] = table_R[RGB[W * H - ((j) / (W) + 1) * W + (j) % (W)]];
						index++;
					}
				}

			} else {		
				index = 0;
				for (j = 0; j < W * H; j++) {
					if ((j % W < W - 1 && Info_h.biWidth % 2) || !(Info_h.biWidth % 2)) {
						unsigned char pixel_index = RGB[W * H - ((j) / (W) + 1) * W + (j) % (W)];
						image->comps[0].data[index] = table_R[pixel_index];
						image->comps[1].data[index] = table_G[pixel_index];
						image->comps[2].data[index] = table_B[pixel_index];
						index++;
					}
				}
			}
			free(RGB);
      free(table_R);
      free(table_G);
      free(table_B);
		} else if (Info_h.biBitCount == 8 && Info_h.biCompression == 1) {				
			table_R = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_G = (unsigned char *) malloc(256 * sizeof(unsigned char));
			table_B = (unsigned char *) malloc(256 * sizeof(unsigned char));
			
			for (j = 0; j < Info_h.biClrUsed; j++) {
				table_B[j] = getc(IN);
				table_G[j] = getc(IN);
				table_R[j] = getc(IN);
				getc(IN);
				if (table_R[j] != table_G[j] && table_R[j] != table_B[j] && table_G[j] != table_B[j])
					gray_scale = 0;
			}

			numcomps = gray_scale ? 1 : 3;
			color_space = gray_scale ? CLRSPC_GRAY : CLRSPC_SRGB;
			/* initialize image components */
			memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
			for(i = 0; i < numcomps; i++) {
				cmptparm[i].prec = 8;
				cmptparm[i].bpp = 8;
				cmptparm[i].sgnd = 0;
				cmptparm[i].dx = subsampling_dx;
				cmptparm[i].dy = subsampling_dy;
				cmptparm[i].w = w;
				cmptparm[i].h = h;
			}
			/* create the image */
			image = opj_image_create(numcomps, &cmptparm[0], color_space);
			if(!image) {
				fclose(IN);
				return NULL;
			}

			/* set image offset and reference grid */
			image->x0 = parameters->image_offset_x0;
			image->y0 = parameters->image_offset_y0;
			image->x1 =	!image->x0 ? (w - 1) * subsampling_dx + 1 : image->x0 + (w - 1) * subsampling_dx + 1;
			image->y1 =	!image->y0 ? (h - 1) * subsampling_dy + 1 : image->y0 + (h - 1) * subsampling_dy + 1;

			/* set image data */
			
			/* Place the cursor at the beginning of the image information */
			fseek(IN, 0, SEEK_SET);
			fseek(IN, File_h.bfOffBits, SEEK_SET);
			
			RGB = (unsigned char *) malloc(Info_h.biWidth * Info_h.biHeight * sizeof(unsigned char));
            
			while (not_end_file) {
				v = getc(IN);
				if (v) {
					v2 = getc(IN);
					for (i = 0; i < (int) v; i++) {
						RGB[line * Info_h.biWidth + col] = v2;
						col++;
					}
				} else {
					v = getc(IN);
					switch (v) {
						case 0:
							col = 0;
							line++;
							break;
						case 1:
							line++;
							not_end_file = 0;
							break;
						case 2:
							fprintf(stderr,"No Delta supported\n");
							opj_image_destroy(image);
							fclose(IN);
							return NULL;
						default:
							for (i = 0; i < v; i++) {
								v2 = getc(IN);
								RGB[line * Info_h.biWidth + col] = v2;
								col++;
							}
							if (v % 2)
								v2 = getc(IN);
							break;
					}
				}
			}
			if (gray_scale) {
				index = 0;
				for (line = 0; line < Info_h.biHeight; line++) {
					for (col = 0; col < Info_h.biWidth; col++) {
						image->comps[0].data[index] = table_R[(int)RGB[(Info_h.biHeight - line - 1) * Info_h.biWidth + col]];
						index++;
					}
				}
			} else {
				index = 0;
				for (line = 0; line < Info_h.biHeight; line++) {
					for (col = 0; col < Info_h.biWidth; col++) {
						unsigned char pixel_index = (int)RGB[(Info_h.biHeight - line - 1) * Info_h.biWidth + col];
						image->comps[0].data[index] = table_R[pixel_index];
						image->comps[1].data[index] = table_G[pixel_index];
						image->comps[2].data[index] = table_B[pixel_index];
						index++;
					}
				}
			}
			free(RGB);
      free(table_R);
      free(table_G);
      free(table_B);
	} else {
		fprintf(stderr, 
			"Other system than 24 bits/pixels or 8 bits (no RLE coding) is not yet implemented [%d]\n", Info_h.biBitCount);
	}
	fclose(IN);
 }
 
 return image;
}


unsigned char readuchar(FILE * f)
{
  unsigned char c1;
  size_t size = fread(&c1, 1, 1, f);
  return c1;
}

unsigned short readushort(FILE * f, int bigendian)
{
  unsigned char c1, c2;
  size_t size = fread(&c1, 1, 1, f);
  size = fread(&c2, 1, 1, f);
  if (bigendian)
    return (c1 << 8) + c2;
  else
    return (c2 << 8) + c1;
}

unsigned int readuint(FILE * f, int bigendian)
{
  unsigned char c1, c2, c3, c4;
  size_t size = fread(&c1, 1, 1, f);
  size = fread(&c2, 1, 1, f);
  size = fread(&c3, 1, 1, f);
  size = fread(&c4, 1, 1, f);
  if (bigendian)
    return (c1 << 24) + (c2 << 16) + (c3 << 8) + c4;
  else
    return (c4 << 24) + (c3 << 16) + (c2 << 8) + c1;
}

opj_image_t*
opj_pgx_to_image(const char *filename, opj_cparameters_t *parameters)
{
	FILE *f = NULL;
	int w, h, prec;
	int i, numcomps, max;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t cmptparm;	/* maximum of 1 component  */
	opj_image_t * image = NULL;

	char endian1,endian2,sign;
	char signtmp[32];

	char temp[32];
	int bigendian;
	opj_image_comp_t *comp = NULL;

	numcomps = 1;
	color_space = CLRSPC_GRAY;

	memset(&cmptparm, 0, sizeof(opj_image_cmptparm_t));

	max = 0;

	f = fopen(filename, "rb");
	if (!f) {
	  fprintf(stderr, "Failed to open %s for reading !\n", filename);
	  return NULL;
	}

	fseek(f, 0, SEEK_SET);
	size_t size = fscanf(f, "PG%[ \t]%c%c%[ \t+-]%d%[ \t]%d%[ \t]%d",temp,&endian1,&endian2,signtmp,&prec,temp,&w,temp,&h);
	
	i=0;
	sign='+';		
	while (signtmp[i]!='\0') {
		if (signtmp[i]=='-') sign='-';
		i++;
	}
	
	fgetc(f);
	if (endian1=='M' && endian2=='L') {
		bigendian = 1;
	} else if (endian2=='M' && endian1=='L') {
		bigendian = 0;
	} else {
		fprintf(stderr, "Bad pgx header, please check input file\n");
		return NULL;
	}

	/* initialize image component */

	cmptparm.x0 = parameters->image_offset_x0;
	cmptparm.y0 = parameters->image_offset_y0;
	cmptparm.w = !cmptparm.x0 ? (w - 1) * parameters->subsampling_dx + 1 : cmptparm.x0 + (w - 1) * parameters->subsampling_dx + 1;
	cmptparm.h = !cmptparm.y0 ? (h - 1) * parameters->subsampling_dy + 1 : cmptparm.y0 + (h - 1) * parameters->subsampling_dy + 1;
	
	if (sign == '-') {
		cmptparm.sgnd = 1;
	} else {
		cmptparm.sgnd = 0;
	}
	cmptparm.prec = prec;
	cmptparm.bpp = prec;
	cmptparm.dx = parameters->subsampling_dx;
	cmptparm.dy = parameters->subsampling_dy;
	
	/* create the image */
	image = opj_image_create(numcomps, &cmptparm, color_space);
	if(!image) {
		fclose(f);
		return NULL;
	}
	/* set image offset and reference grid */
	image->x0 = cmptparm.x0;
	image->y0 = cmptparm.x0;
	image->x1 = cmptparm.w;
	image->y1 = cmptparm.h;

	/* set image data */

	comp = &image->comps[0];

	for (i = 0; i < w * h; i++) {
		int v;
		if (comp->prec <= 8) {
			if (!comp->sgnd) {
				v = readuchar(f);
			} else {
				v = (char) readuchar(f);
			}
		} else if (comp->prec <= 16) {
			if (!comp->sgnd) {
				v = readushort(f, bigendian);
			} else {
				v = (short) readushort(f, bigendian);
			}
		} else {
			if (!comp->sgnd) {
				v = readuint(f, bigendian);
			} else {
				v = (int) readuint(f, bigendian);
			}
		}
		if (v > max)
			max = v;
		comp->data[i] = v;
	}
	fclose(f);
	comp->bpp = int_floorlog2(max) + 1;

	return image;
}

opj_image_t*
opj_pnm_to_image(const char *filename, opj_cparameters_t *parameters)
{
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;

	FILE *f = NULL;
	int i, compno, numcomps, w, h;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t cmptparm[3];	/* maximum of 3 components */
	opj_image_t * image = NULL;
	char value;
	
	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open %s for reading !!\n", filename);
		return 0;
	}

	if (fgetc(f) != 'P')
		return 0;
	value = fgetc(f);

		switch(value) {
			case '2':	/* greyscale image type */
			case '5':
				numcomps = 1;
				color_space = CLRSPC_GRAY;
				break;
				
			case '3':	/* RGB image type */
			case '6':
				numcomps = 3;
				color_space = CLRSPC_SRGB;
				break;
				
			default:
				fclose(f);
				return NULL;
		}
		
		fgetc(f);
		
		/* skip comments */
		while(fgetc(f) == '#') while(fgetc(f) != '\n');
		
		fseek(f, -1, SEEK_CUR);
		size_t size = fscanf(f, "%d %d\n255", &w, &h);			
		fgetc(f);	/* <cr><lf> */
		
	/* initialize image components */
	memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
	for(i = 0; i < numcomps; i++) {
		cmptparm[i].prec = 8;
		cmptparm[i].bpp = 8;
		cmptparm[i].sgnd = 0;
		cmptparm[i].dx = subsampling_dx;
		cmptparm[i].dy = subsampling_dy;
		cmptparm[i].w = w;
		cmptparm[i].h = h;
	}
	/* create the image */
	image = opj_image_create(numcomps, &cmptparm[0], color_space);
	if(!image) {
		fclose(f);
		return NULL;
	}

	/* set image offset and reference grid */
	image->x0 = parameters->image_offset_x0;
	image->y0 = parameters->image_offset_y0;
	image->x1 = parameters->image_offset_x0 + (w - 1) *	subsampling_dx + 1;
	image->y1 = parameters->image_offset_y0 + (h - 1) *	subsampling_dy + 1;

	/* set image data */

	if ((value == '2') || (value == '3')) {	/* ASCII */
		for (i = 0; i < w * h; i++) {
			for(compno = 0; compno < numcomps; compno++) {
				unsigned int index = 0;
				size_t size = fscanf(f, "%u", &index);
				/* compno : 0 = GREY, (0, 1, 2) = (R, G, B) */
				image->comps[compno].data[i] = index;
			}
		}
	} else if ((value == '5') || (value == '6')) {	/* BINARY */
		for (i = 0; i < w * h; i++) {
			for(compno = 0; compno < numcomps; compno++) {
				unsigned char index = 0;
				size_t size = fread(&index, 1, 1, f);
				/* compno : 0 = GREY, (0, 1, 2) = (R, G, B) */
				image->comps[compno].data[i] = index;
			}
		}
	}

	fclose(f);

	return image;
}

opj_image_t*
opj_raw_to_image(const char *filename, opj_cparameters_t *parameters, raw_cparameters_t *raw_cp)
{
	int subsampling_dx = parameters->subsampling_dx;
	int subsampling_dy = parameters->subsampling_dy;

	FILE *f = NULL;
	int i, compno, numcomps, w, h;
	OPJ_COLOR_SPACE color_space;
	opj_image_cmptparm_t *cmptparm;	
	opj_image_t * image = NULL;
	unsigned short ch;
	
	if((! (raw_cp->rawWidth & raw_cp->rawHeight & raw_cp->rawComp & raw_cp->rawBitDepth)) == 0)
	{
		fprintf(stderr,"\nError: invalid raw image parameters\n");
		fprintf(stderr,"Please use the Format option -F:\n");
		fprintf(stderr,"-F rawWidth,rawHeight,rawComp,rawBitDepth,s/u (Signed/Unsigned)\n");
		fprintf(stderr,"Example: -i lena.raw -o lena.j2k -F 512,512,3,8,u\n");
		fprintf(stderr,"Aborting\n");
		return NULL;
	}

	f = fopen(filename, "rb");
	if (!f) {
		fprintf(stderr, "Failed to open %s for reading !!\n", filename);
		fprintf(stderr,"Aborting\n");
		return NULL;
	}
	numcomps = raw_cp->rawComp;
	color_space = CLRSPC_SRGB;
	w = raw_cp->rawWidth;
	h = raw_cp->rawHeight;
	cmptparm = (opj_image_cmptparm_t*) malloc(numcomps * sizeof(opj_image_cmptparm_t));
	
	/* initialize image components */	
	memset(&cmptparm[0], 0, numcomps * sizeof(opj_image_cmptparm_t));
	for(i = 0; i < numcomps; i++) {		
		cmptparm[i].prec = raw_cp->rawBitDepth;
		cmptparm[i].bpp = raw_cp->rawBitDepth;
		cmptparm[i].sgnd = raw_cp->rawSigned;
		cmptparm[i].dx = subsampling_dx;
		cmptparm[i].dy = subsampling_dy;
		cmptparm[i].w = w;
		cmptparm[i].h = h;
	}
	/* create the image */
	image = opj_image_create(numcomps, &cmptparm[0], color_space);
	if(!image) {
		fclose(f);
		return NULL;
	}
	/* set image offset and reference grid */
	image->x0 = parameters->image_offset_x0;
	image->y0 = parameters->image_offset_y0;
	image->x1 = parameters->image_offset_x0 + (w - 1) *	subsampling_dx + 1;
	image->y1 = parameters->image_offset_y0 + (h - 1) *	subsampling_dy + 1;

	if(raw_cp->rawBitDepth <= 8)
	{
		unsigned char value = 0;
		for(compno = 0; compno < numcomps; compno++) {
			for (i = 0; i < w * h; i++) {
				if (!fread(&value, 1, 1, f)) {
					fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
					return NULL;
				}
				image->comps[compno].data[i] = raw_cp->rawSigned?(char)value:value;
			}
		}
	}
	else if(raw_cp->rawBitDepth <= 16)
	{
		unsigned short value;
		for(compno = 0; compno < numcomps; compno++) {
			for (i = 0; i < w * h; i++) {
				unsigned char temp;
				if (!fread(&temp, 1, 1, f)) {
					fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
					return NULL;
				}
				value = temp << 8;
				if (!fread(&temp, 1, 1, f)) {
					fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
					return NULL;
				}
				value += temp;
				image->comps[compno].data[i] = raw_cp->rawSigned?(short)value:value;
			}
		}
	}
	else {
		fprintf(stderr,"OpenJPEG cannot encode raw components with bit depth higher than 16 bits.\n");
		return NULL;
	}

	if (fread(&ch, 1, 1, f)) {
		fprintf(stderr,"Warning. End of raw file not reached... processing anyway\n");
	}
	fclose(f);

	return image;
}
