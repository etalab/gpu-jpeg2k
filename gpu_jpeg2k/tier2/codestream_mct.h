/**
 * @file codestream_mct.h
 *
 * @author Kamil Balwierz
 */
#ifndef CODESTREAM_MCT_H_
#define CODESTREAM_MCT_H_

void write_multiple_component_transformations(type_buffer *buffer, type_image *img);
void read_multiple_component_transformations(type_buffer *buffer, type_image *img);

#endif
