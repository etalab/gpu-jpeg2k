#
# OpenJPEG Library MAKEFILE
#
# Library is extended by ability to load BMP image which is implemented
# in files opj_bmp.c and opj_bmp.h
#

# Configure
TARGET := openjpeg.a
CFILES := bio.c cio.c dwt.c event.c image.c j2k.c j2k_lib.c jp2.c jpt.c mct.c \
          mqc.c openjpeg.c pi.c raw.c t1.c t2.c tcd.c tgt.c opj_convert.c
#CFLAGS   += -D_DATA_HOOK_DISABLED

# Include Common Makefile
include ../../../common.mk

# Change optimalization level
OPTFLAGS := -O3

# Headers for files
bio.c.o: bio.c bio.h opj_includes.h
cio.c.o: cio.c cio.h opj_includes.h
dwt.c.o: dwt.c dwt.h opj_includes.h
event.c.o: event.c event.h opj_includes.h
image.c.o: image.c image.h opj_includes.h
j2k.c.o: j2k.c j2k.h opj_includes.h
j2k_lib.c.o: j2k_lib.c j2k_lib.h opj_includes.h
jp2.c.o: jp2.c jp2.h opj_includes.h
jpt.c.o: jpt.c jpt.h opj_includes.h
mct.c.o: mct.c mct.h opj_includes.h
mqc.c.o: mqc.c mqc.h opj_includes.h
openjpeg.c.o: openjpeg.c openjpeg.h opj_includes.h
pi.c.o: pi.c pi.h opj_includes.h
raw.c.o: raw.c raw.h opj_includes.h
t1.c.o: t1.c t1.h opj_includes.h
t2.c.o: t2.c t2.h opj_includes.h
tcd.c.o: tcd.c tcd.h opj_includes.h
tgt.c.o: tgt.c tgt.h opj_includes.h
opj_convert.c.o: opj_convert.c opj_convert.h
