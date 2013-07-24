/* 
Copyright 2009-2013 Poznan Supercomputing and Networking Center

Authors:
Milosz Ciznicki miloszc@man.poznan.pl

GPU JPEG2K is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GPU JPEG2K is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with GPU JPEG2K. If not, see <http://www.gnu.org/licenses/>.
*/
/**
 * @file allocate_memory.h
 *
 * @author Milosz Ciznicki
 */


#ifndef ALLOCATE_MEM_H_
#define ALLOCATE_MEM_H_

#include <stdint.h>
#include <stddef.h>

extern void cuda_set_device_flags();
extern void cuda_h_allocate_mem(void **data, uint64_t memSize);
extern void cuda_d_allocate_mem(void **data, uint64_t memSize);
extern void cuda_memcpy_hth(void *src, void *dst, uint64_t size);
extern void cuda_memcpy_htd(void *src, void *dst, uint64_t size);
extern void cuda_memcpy_dtd(void *src, void *dst, uint64_t size);
extern void cuda_memcpy_dth(void *src, void *dst, uint64_t size);
void cuda_memcpy2d_dtd(void *src, size_t src_width, void *dst, size_t dst_width, size_t width, size_t height);
void cuda_memcpy2d_dth(void *src, size_t src_width, void *dst, size_t dst_width, size_t width, size_t height);
void cuda_memcpy2d_htd(void *src, size_t src_width, void *dst, size_t dst_width, size_t width, size_t height);
extern void cuda_h_free(void *data);
extern void cuda_d_free(void *data);
extern void cuda_d_memset(void *dst, int val, uint64_t mem_size);
extern void cuda_set_device(int i);
extern void cuda_set_printf_limit(size_t memSize);

#endif /* ALLOCATE_MEM_H_ */
