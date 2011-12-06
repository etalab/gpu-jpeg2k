#include "file_access.h"
#include "../print_info/print_info.h"

#include <stdlib.h>

//int write_byte(FILE *fd, unsigned char value) {
//	fputc(value, fd);
///*
//	if(fputc(value, fd) == EOF){
//		printf("Error while writing file!");
//		return(-1);
//	}
//	else
//*/
//	return 0;
//
//}
//
//int write_short(FILE *fd, unsigned short value) {
//	write_byte(fd, value >> 8);
//	write_byte(fd, value);
//    return 0;
//}
//
//int write_int(FILE *fd, unsigned int value) {
//	write_byte(fd, value >> 24);
//	write_byte(fd, value >> 16);
//	write_byte(fd, value >> 8);
//	write_byte(fd, value);
//	return 0;
//}

//unsigned char read_byte(FILE *fd) {
//	return fgetc(fd);
//	//error checking?
//}
//
//unsigned char *read_bytes(unsigned char *dest, FILE *fd, long int n) {
//	long int i;
//	for(i = 0; i< n; i++) {
//		if( (dest[i] = fgetc(fd)) == EOF ) {
//			println(INFO, "This should not happen: EOF");
//			return NULL;
//		}
//	}
//
//
//	dest[i] = '\0';
//
//	printf("in read: %p\n", dest);
//	return (unsigned char *)dest;
//}

//
//unsigned char *read_4bytes(unsigned char *mem, long int *pos) {
//unsigned char *read_4bytes(unsigned char *mem, long int *pos) {811388
//	unsigned char *ret = malloc(4 * sizeof(unsigned char));
//
//	printf("pos: %li\n", *pos);
//	ret[0] = mem[*pos];
//	ret[1] = mem[++(*pos)];
//	ret[2] = mem[++(*pos)];
//	ret[3] = mem[++(*pos)];
//
//
//	return ret;
//}
