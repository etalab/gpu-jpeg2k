#ifndef _MQC_GPU_COMMON_H
#define _MQC_GPU_COMMON_H

// Maximum grid size
#define MAXIMUM_GRID_SIZE 65535

// Max output buffer size for CX,D count
#define cxd_get_buffer_size(cxd_count) (16 + cxd_count / 8)

// CX,D pair manipulation
#define cxd_make(cx,d)  ((unsigned char)((d << 5) | cx))
#define cxd_get_cx(cxd) ((unsigned char)(cxd & 0x1F))
#define cxd_get_d(cxd)  ((unsigned char)(cxd >> 5))

// CX,D array manipulation, one CX,D per byte
#define cxd_array_size(count)          count
#define cxd_array_put(ptr,index,cx,d)  ptr[index] = cxd_make(cx,d)
#define cxd_array_get(ptr,index)       ptr[index]

// CX,D array manipulation, one CX,D per 6 bits
/*
#undef cxd_array_size
#define cxd_array_size(count) \
    (((count * 6) / 8) + 2) \

#undef cxd_array_put
#define cxd_array_put(ptr,index,cx,d) \
    unsigned int __value = ptr[(index * 6) / 8] << 8 | ptr[(index * 6) / 8 + 1]; \
    __value |= ((((unsigned int)cxd_make(cx,d)) << 10) >> ((index * 6) % 8)); \
    ptr[(index * 6) / 8] = (unsigned char)(__value >> 8); \
    ptr[(index * 6) / 8 + 1] = (unsigned char)(__value); \

#undef cxd_array_get
#define cxd_array_get(ptr,index) \
    ((unsigned char)(((((unsigned int)ptr[(index * 6) / 8] << 8 | ptr[(index * 6) / 8 + 1]) >> (10 - (index * 6) % 8))) & 0x3F)) \
*/

// CX,D block
struct cxd_block {
    int cxd_begin;
    int cxd_count;
    int byte_begin;
    int byte_count;
};

#endif
