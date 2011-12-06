#!/bin/bash

BUILD_DIR=/home/miloszc/Projects/gpu_jpeg2k/build
FILENAME=list.txt
OUT_DIR=output

IMG_DIR=${PWD}/${1}/
EXT=${2}

# $1 image directory
# $2 extension of image files

cd ${IMG_DIR}
ls ${EXT} > $FILENAME
cat $FILENAME
mv $FILENAME $BUILD_DIR

cd $BUILD_DIR
make
rm -r $OUT_DIR
mkdir $OUT_DIR

COUNT=0

while read LINE
do
	echo "${LINE}"
	for i in {1..5}
	do
		./encoder -i ${IMG_DIR}${LINE} -o ${LINE}.j2k -c ../lossless.config
	done
	let COUNT++
done < $FILENAME

mv *.j2k ${OUT_DIR}

rm $FILENAME
