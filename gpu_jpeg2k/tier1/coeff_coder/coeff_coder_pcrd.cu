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
/*
 * coeff_coder_pcrd.cu
 *
 *  Created on: Dec 8, 2011
 *      Author: miloszc
 */

#include <stdio.h>
#include "gpu_coeff_coder2.cuh"
#include "gpu_mq-coder.cuh"
extern "C" {
	#include "../../misc/memory_management.cuh"
}

namespace GPU_JPEG2K
{

__global__ void g_slopeCalculation(int codeBlocks, int maxStatesPerCodeBlock, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos, CodeBlockAdditionalInfo *infos, float *d_slope_max)
{
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	if(threadId >= codeBlocks)
//	if(threadId != 1)
		return;

	pcrdCodeblocks += maxStatesPerCodeBlock * threadId;
	pcrdCodeblockInfos += threadId;
	infos += threadId;

	int nFeasible = 0;
	int lastSlope = 0;
	float deltaD, deltaL;

	int nstates = infos->codingPasses;

	pcrdCodeblocks[nFeasible].slope = /*FLT_MAX*/1000000000;
	pcrdCodeblocks[nFeasible].feasiblePoint = 0;

	nFeasible = 1;
	pcrdCodeblockInfos->Lfirst = pcrdCodeblocks[nFeasible].L;

	for(int i = 1; i < nstates; i++)
	{
		deltaD = pcrdCodeblocks[lastSlope].dist - pcrdCodeblocks[i].dist;
		deltaL = (float)(pcrdCodeblocks[i].L - pcrdCodeblocks[lastSlope].L);

//		if(threadId == 12)
//		printf("%f %f deltaD:%f slope*deltaL:%f L[%d]:%d last[%d]:%d deltaL:%f\n", pcrdCodeblocks[lastSlope].dist , pcrdCodeblocks[i].dist, deltaD, pcrdCodeblocks[lastSlope].slope * deltaL, i, pcrdCodeblocks[i].L, lastSlope, pcrdCodeblocks[lastSlope].L, deltaL);

		if(deltaD > 0.0f)
		{
			while(deltaD >= pcrdCodeblocks[lastSlope].slope * deltaL)
			{
				/*if(nFeasible == 0)
				{
					printf("nFeasible:%d tid:%d\n", nFeasible, threadId);
					return;
				}*/
				nFeasible--;
				lastSlope = pcrdCodeblocks[nFeasible - 1].feasiblePoint;
				deltaD = pcrdCodeblocks[lastSlope].dist - pcrdCodeblocks[i].dist;
				deltaL = (float)(pcrdCodeblocks[i].L - pcrdCodeblocks[lastSlope].L);
			}
			lastSlope = i;
			pcrdCodeblocks[nFeasible++].feasiblePoint = lastSlope;
			pcrdCodeblocks[lastSlope].slope = deltaD/deltaL;
		}
	}

	pcrdCodeblockInfos->feasibleNum = nFeasible;

	int feasiblePoint;

	for(int i = 0; i < nFeasible; i++)
	{
		feasiblePoint = pcrdCodeblocks[i].feasiblePoint;
		pcrdCodeblocks[i].L = pcrdCodeblocks[feasiblePoint].L;
		pcrdCodeblocks[i].slope = pcrdCodeblocks[feasiblePoint].slope;

//		printf("%f\n", *d_slope_max);
		if((pcrdCodeblocks[i].slope > *d_slope_max) && (i != 0))
		{
			*d_slope_max = pcrdCodeblocks[i].slope;
		}
//		printf("cblk:%2d pcrdCodeblockInfos->Lfirst:%3d feasiblePoints:%6d truncLengths[i]:%6d slopes:%6f\n", threadId, pcrdCodeblockInfos->Lfirst, pcrdCodeblockInfos->feasibleNum, i, pcrdCodeblocks[i].L, pcrdCodeblocks[i].slope);
	}
}

#define CBLKS_PER_THREAD 16
#define THREADS_PER_BLOCK 4

__global__ void g_truncateSize(int codeBlocks, int maxStatesPerCodeBlock, CodeBlockAdditionalInfo *infos, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos, float lambda, int *sizes)
{
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	int cbStart = threadId * CBLKS_PER_THREAD;
	int cbStop = (threadId + 1) * CBLKS_PER_THREAD > codeBlocks ? codeBlocks : (threadId + 1) * CBLKS_PER_THREAD;
	int i, size = 0;

//	printf("cbStart:%d cbStop:%d\n", cbStart, cbStop);

	for(i = cbStart; i < cbStop; ++i)
	{
		CodeBlockAdditionalInfo * info = &(infos[i]);
		PcrdCodeblock *pcrdCodeblock = &(pcrdCodeblocks[maxStatesPerCodeBlock * i]);
		PcrdCodeblockInfo *pcrdCodeblockInfo = &(pcrdCodeblockInfos[i]);
		/*infos += i;
		pcrdCodeblocks += maxStatesPerCodeBlock * i;
		pcrdCodeblockInfos += i;*/

		int j = 0;

		while((j + 1 < pcrdCodeblockInfo->feasibleNum) && (pcrdCodeblock[j + 1].slope > lambda)) j++;

		if(j > 0)
		{
			info->codingPasses = pcrdCodeblock[j].feasiblePoint;
			info->length = pcrdCodeblock[j].L;
			size += pcrdCodeblock[j].L;
//			printf("j > 0 cblk:%d size:%d\n", i, size);
		} else
		{
			info->codingPasses = 1;
			info->length = pcrdCodeblockInfo->Lfirst;
			size += pcrdCodeblockInfo->Lfirst;
//			printf("j <= 0 cblk:%d pcrdCodeblockInfos->Lfirst:%d size:%d\n", i, pcrdCodeblockInfo->Lfirst, size);
		}
	}

	if(cbStart < codeBlocks)
		sizes[threadId] = size;
}

int truncateSize(int codeBlocks, int maxStatesPerCodeBlock, CodeBlockAdditionalInfo *infos, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos, float lambda)
{
	int threads = (codeBlocks + CBLKS_PER_THREAD - 1) / CBLKS_PER_THREAD;
	int blocks = (threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int size = 0;

	int *sizes_h, *sizes_d;

	cuda_d_allocate_mem((void **)&sizes_d, sizeof(int) * threads);
	cuda_d_memset(sizes_d, 0, sizeof(int) * threads);
	cuda_h_allocate_mem((void **)&sizes_h, sizeof(int) * threads);

//	printf("threads:%d blocks:%d\n", threads, blocks);

	g_truncateSize<<<blocks, THREADS_PER_BLOCK>>>(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambda, sizes_d);

	cudaThreadSynchronize();

	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", "g_truncateSize", cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

	cuda_memcpy_dth(sizes_d, sizes_h, sizeof(int) * threads);

	int i;

	for(i = 0; i < threads; ++i)
	{
//		printf("sizes_h[%d]:%d\n", i, sizes_h[i]);
		size += sizes_h[i];
	}

	cuda_d_free(sizes_d);
	cuda_h_free(sizes_h);

	return size;
}

#define ALLOWED_DIFF 400.0f

void launch_pcrd(int maxStatesPerCodeBlock, int targetSize, float slopeMax, int codeBlocks, CodeBlockAdditionalInfo *infos, PcrdCodeblock *pcrdCodeblocks, PcrdCodeblockInfo *pcrdCodeblockInfos)
{
	float lambdaMin = -1.0f;
	float lambdaMax = slopeMax * 2.0f;
	float lambdaMid;
	int overHead, minSize, maxSize;

	//TODO
	overHead = 181 + /*numTiles *//*3 * 14 + */codeBlocks * 2;
//	printf("over head:%d\n", overHead);
//	overHead = 0;

	minSize = overHead + truncateSize(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambdaMax);

//	printf("minSize:%d\n", minSize);

	if(targetSize <= minSize)
	{
		printf("Target size to small %d using %d\n", targetSize, minSize);
		return;
	}

	maxSize = overHead + truncateSize(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambdaMin);

//	printf("maxSize:%d\n", maxSize);

	if(targetSize >= maxSize)
	{
		printf("Target size to large %d using %d\n", targetSize, maxSize);
		return;
	}

	float allowedDiff = ALLOWED_DIFF > (0.02f * (float)targetSize) ? ALLOWED_DIFF : (0.02f * (float)targetSize);
	int size, iterations = 0, countRefine = 0;

	do {
		lambdaMid = 0.5f * (lambdaMin + lambdaMax);

		size = overHead + truncateSize(codeBlocks, maxStatesPerCodeBlock, infos, pcrdCodeblocks, pcrdCodeblockInfos, lambdaMid);

//		printf("size:%d\n", size);

		if(size < targetSize)
			lambdaMax = lambdaMid;
		else
			lambdaMin = lambdaMid;

		if(countRefine == 0)
		{
			if(abs(targetSize - size) < allowedDiff)
			{
				countRefine = 1;
//				printf("allowedDiff:%d\n", abs(targetSize - size));
			}
		} else
		{
			countRefine++;
		}

		iterations++;
	} while(countRefine < 20 && iterations < 50);
}

void launch_encode_pcrd(dim3 gridDim, dim3 blockDim, CoefficientState *coeffBuffors, byte *outbuf, int maxThreadBufforLength, CodeBlockAdditionalInfo *infos, int codeBlocks, int targetSize)
{
	const int maxMQStatesPerCodeBlock = (MAX_MAG_BITS - 1) * 3 + 1;

	PcrdCodeblockInfo *pcrdCodeblockInfos;
	cuda_d_allocate_mem((void **) &pcrdCodeblockInfos, sizeof(PcrdCodeblockInfo) * codeBlocks);

	PcrdCodeblock *pcrdCodeblocks;
	cuda_d_allocate_mem((void **) &pcrdCodeblocks, sizeof(PcrdCodeblock) * codeBlocks * maxMQStatesPerCodeBlock);
	cuda_d_memset((void *)pcrdCodeblocks, 0, sizeof(PcrdCodeblock) * codeBlocks * maxMQStatesPerCodeBlock);

	_launch_encode_pcrd(gridDim, blockDim, coeffBuffors, outbuf, maxThreadBufforLength, infos, codeBlocks, maxMQStatesPerCodeBlock, pcrdCodeblocks, pcrdCodeblockInfos);

	float *dSlopeMax;
	cuda_d_allocate_mem((void**)&dSlopeMax, sizeof(float));
	cuda_d_memset((void *)dSlopeMax, 0, sizeof(float));

	g_slopeCalculation<<<(int) ceil(codeBlocks / 512.0f), 512>>>(codeBlocks, maxMQStatesPerCodeBlock, pcrdCodeblocks, pcrdCodeblockInfos, infos, dSlopeMax);

	//TODO debug
	PcrdCodeblockInfo *pcrdCodeblockInfos_h;
	pcrdCodeblockInfos_h = (PcrdCodeblockInfo *) my_malloc(sizeof(PcrdCodeblockInfo) * codeBlocks);
	cuda_memcpy_dth(pcrdCodeblockInfos, pcrdCodeblockInfos_h, sizeof(PcrdCodeblockInfo) * codeBlocks);

	/*for(int i = 0; i < codeBlocks; ++i)
	{
		printf("cblk:%d feasibleNum:%d Lfirst:%d\n", i, pcrdCodeblockInfos_h[i].feasibleNum, pcrdCodeblockInfos_h[i].Lfirst);
	}*/

	float slopeMax;
	cuda_memcpy_dth(dSlopeMax, &slopeMax, sizeof(float));

	//TODO
//	int targetSize = 380;

	launch_pcrd(maxMQStatesPerCodeBlock, targetSize, slopeMax, codeBlocks, infos, pcrdCodeblocks, pcrdCodeblockInfos);

	cuda_d_free(pcrdCodeblocks);
	cuda_d_free(pcrdCodeblockInfos);

//	launch_pcrd(infos, codeBlocks, mqstates, maxMQStatesPerCodeBlock, pcrd);
}

}
