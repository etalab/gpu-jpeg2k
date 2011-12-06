/**
 * @file image_bil.c
 *
 * @author Milosz Ciznicki
 */
#include "../types/image_bil.h"
#include "../print_info/print_info.h"
#include <FreeImage.h>

/**
 * @brief Initializes image container
 *
 * @param container Malloc'ed type_image to be initialized.
 * @param image_bil Image properties.
 */
void init_bil_image(type_image *container, type_image_hyper *image_bil, type_parameters *param)
{
	println_start(INFO);
	container->height = image_bil->num_bands;
	container->width = image_bil->num_samples;
	container->depth = (image_bil->data_type == 1 ? 8 : 16) * image_bil->num_bands;
	container->num_components = image_bil->num_lines;
	container->num_range_bits = container->depth / image_bil->num_bands;
	printf("%d %d %d %d %d\n", container->height, container->width, container->depth, container->num_components, container->num_range_bits);

	/* TODO: check if origin data was signed? */
	container->sign = SIGNED;
	container->num_dlvls = param->param_tile_comp_dlvls;

	/* Set coding parameters */
	set_coding_parameters(container, param);
	init_tiles(&container, param);
}

/**
 * @brief Loads the hyperspectral data.
 *
 * @param image_filename
 * @param image_bil
 * @return
 */
static float *load_image(const char *image_filename, type_image_hyper *image_bil)
{
	println_start(INFO);
	FILE *fp;
	short *type_short_int;
	double *type_double;
	float *type_float;
	float *h_imagen;
	int i;

	h_imagen = (float*) malloc(image_bil->num_lines * image_bil->num_samples * image_bil->num_bands * sizeof(float));

	if (strstr(image_filename, ".bil") == NULL) {
		printf("ERROR: Can not find file bil: %s\n", image_filename);
		exit(1);
	}
	if ((fp = fopen(image_filename, "rb")) == NULL) {
		printf("ERROR %d. Can not open file: %s\n", errno, image_filename);
		exit(1);
	} else {
		fseek(fp, 0L, SEEK_SET);
		switch (image_bil->data_type) {
		case 1: {
			int compno = 0;
			for(compno = 0; compno < image_bil->num_bands; compno++) {
				for (i = 0; i < image_bil->num_lines * image_bil->num_samples; i++) {
					unsigned char temp;
					if (!fread(&temp, 1, 1, fp)) {
						fprintf(stderr,"Error reading raw file. End of file probably reached.\n");
						return NULL;
					}
					h_imagen[i] = ((char)temp);
				}
			}
		}
		break;
		case 2: {
			type_short_int = (short *) malloc(image_bil->num_lines * image_bil->num_samples * image_bil->num_bands
					* sizeof(short));

			int read_lines = fread(type_short_int, sizeof(short), (image_bil->num_lines * image_bil->num_samples * image_bil->num_bands), fp);

			printf("read:%d orginal:%d\n", read_lines, image_bil->num_lines * image_bil->num_samples * image_bil->num_bands);

			for (i = 0; i < image_bil->num_lines * image_bil->num_samples * image_bil->num_bands; i++) {
				h_imagen[i] = (float) type_short_int[i];
			}
			free(type_short_int);
			break;
		}
		case 4: {
			type_float = (float *) malloc(image_bil->num_lines * image_bil->num_samples * image_bil->num_bands
					* sizeof(float));
			fread(type_float, sizeof(float), (sizeof(float) * image_bil->lines_samples * image_bil->num_bands), fp);

			for (i = 0; i < image_bil->num_lines * image_bil->num_samples * image_bil->num_bands; i++) {
				h_imagen[i] = type_float[i];
			}
			free(type_float);
			break;
		}
		case 5: {
			type_double = (double *) malloc(image_bil->num_lines * image_bil->num_samples * image_bil->num_bands
					* sizeof(double));
			fread(type_double, sizeof(double), (sizeof(double) * image_bil->lines_samples * image_bil->num_bands), fp);

			for (i = 0; i < image_bil->num_lines * image_bil->num_samples * image_bil->num_bands; i++) {
				h_imagen[i] = (float) type_double[i];
			}
			free(type_double);
			break;
		}
		}
		fclose(fp);
	}
	return h_imagen;
}

/**
 * @brief Read bsq image from file.
 *
 * @param path Path to hdr image file.
 * @param path Path to bsq image file.
 * @param _container Pointer to a malloc'ed memory for type_image.
 */
type_data *read_bil_image(type_image *container, type_image_hyper *image_bil, type_parameters *param)
{
	init_bil_image(container, image_bil, param);

	return load_image(container->in_file, image_bil);
}
