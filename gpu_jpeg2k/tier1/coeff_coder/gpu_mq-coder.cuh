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
#ifndef GPU_MQ_CODER_CUH_
#define GPU_MQ_CODER_CUH_

#include "gpu_c2luts.cuh"

typedef unsigned char byte;

#define CX_RUN 18
#define CX_UNI 17

struct MQEncoder
{
	short L;

	unsigned short A;
	unsigned int C;
	byte CT;
	byte T;

	byte *outbuf;

	int CXMPS;
	unsigned char CX;

	unsigned int Ib0;
	unsigned int Ib1;
	unsigned int Ib2;
	unsigned int Ib3;
	unsigned int Ib4;
	unsigned int Ib5;
};

struct MQDecoder : public MQEncoder
{
	byte NT;
	int Lmax;
};

__device__ unsigned char getI(MQEncoder &enc, int id)
{
	unsigned char out = 0;

	out |= ((enc.Ib0 >> id) & 1);
	out |= ((enc.Ib1 >> id) & 1) << 1;
	out |= ((enc.Ib2 >> id) & 1) << 2;
	out |= ((enc.Ib3 >> id) & 1) << 3;
	out |= ((enc.Ib4 >> id) & 1) << 4;
	out |= ((enc.Ib5 >> id) & 1) << 5;

	return out;
}

__device__ void setI(MQEncoder &enc, int id, unsigned char value)
{
	unsigned int mask = ~(1 << id);

	enc.Ib0 = (enc.Ib0 & mask) | (((value) & 1) << id);
	enc.Ib1 = (enc.Ib1 & mask) | (((value >> 1) & 1) << id);
	enc.Ib2 = (enc.Ib2 & mask) | (((value >> 2) & 1) << id);
	enc.Ib3 = (enc.Ib3 & mask) | (((value >> 3) & 1) << id);
	enc.Ib4 = (enc.Ib4 & mask) | (((value >> 4) & 1) << id);
	enc.Ib5 = (enc.Ib5 & mask) | (((value >> 5) & 1) << id);
}

__device__ void SwitchNthBit(int &reg, int n)
{
	reg = (reg ^ (1 << n));
}

__device__ short GetNthBit(int &reg, int n)
{
	return (reg >> n) & 1;
}

__device__ void coderResetInd(unsigned char indices[])
{
	#pragma unroll
	for(int i = 1; i < 19; i++)
		indices[i] = 0;

	indices[CX_UNI] = 46;
	indices[CX_RUN] = 3;
	indices[0] = 4;
}

__device__ void mqResetEnc(MQEncoder &encoder)
{
	encoder.Ib0 = 0;
	encoder.Ib1 = 0;
	encoder.Ib2 = 0;
	encoder.Ib3 = 0;
	encoder.Ib4 = 0;
	encoder.Ib5 = 0;

	setI(encoder, CX_UNI, 46);
	setI(encoder, CX_RUN, 3);
	setI(encoder, 0, 4);

	encoder.CXMPS = 0;
}

__device__ void mqResetDec(MQDecoder &decoder)
{
	mqResetEnc(decoder);
}

__device__ void mqInitEnc(MQEncoder &encoder, byte *outbuf)
{
	encoder.A = 0x8000;
	encoder.C = 0;
	encoder.CT = 12;
	encoder.L = -1;
	encoder.outbuf = outbuf;
	encoder.T = 0;
}

//#define BYTEOUT_OPTIMIZED

#ifndef BYTEOUT_OPTIMIZED
__device__ void putByte(MQEncoder &encoder)
{
	if(encoder.L >= 0)
		encoder.outbuf[encoder.L] = encoder.T;
	encoder.L++;
}

__device__ void nonStuffedOut(MQEncoder &encoder)
{
	putByte(encoder);
	encoder.T = encoder.C >> 19;
	encoder.C = encoder.C & 0x7FFFF;
	encoder.CT = 8;
}

__device__ void stuffedOut(MQEncoder &encoder)
{
	putByte(encoder);
	encoder.T = encoder.C >> 20;
	encoder.C = encoder.C & 0xFFFFF;
	encoder.CT = 7;
}
#endif

__device__ void byteout(MQEncoder &encoder)
{
#ifndef BYTEOUT_OPTIMIZED
// alternative version
	if(encoder.T == (unsigned char) 0xFF)
		stuffedOut(encoder);
	else if(encoder.C < 0x8000000)
		nonStuffedOut(encoder);
	else
	{
		encoder.T++;
		if(encoder.T == (unsigned char) 0xFF)
		{
			encoder.C = encoder.C & 0x7FFFFFF;
			stuffedOut(encoder);
		}
		else
			nonStuffedOut(encoder);
	}
#endif

#ifdef BYTEOUT_OPTIMIZED
	int bitstuff;
	int fullbuf = encoder.outbuf[0] == (byte) 0xFF;
	int fullbuf1 = encoder.outbuf[1] == (byte) 0xFF;
	int half_C = encoder.C < 0x8000000;

	bitstuff = fullbuf | ((1 - half_C) & fullbuf1);
	encoder.C &= (0xFFFFFFFF - 0xF8000000 * (1 - fullbuf) * (1 - half_C) * fullbuf1);
	encoder.outbuf[0] += (1 - fullbuf) * (1 - half_C);

	encoder.outbuf += 1;
	encoder.L += 1;
	encoder.outbuf[0] = encoder.C >> (19 + bitstuff);
	encoder.C = encoder.C & (0x7FFFF + 0x80000 * bitstuff);
	encoder.CT = (8 - bitstuff);
#endif
}

__device__ void renorme(MQEncoder &encoder)
{
	do
	{
		encoder.A = encoder.A << 1;
		encoder.C = encoder.C << 1;
		encoder.CT = encoder.CT - 1;

		if(encoder.CT == 0)
			byteout(encoder);
	}
	while((encoder.A & 0x8000) == 0);
}

__device__ void codemps(MQEncoder &encoder)
{
	unsigned int p = Qe[getI(encoder, encoder.CX)];
	encoder.A = encoder.A - p;

	if((encoder.A & 0x8000) == 0)
	{
		if(encoder.A < p)
			encoder.A = p;
		else
			encoder.C = encoder.C + p;

		setI(encoder, encoder.CX, NMPS[getI(encoder, encoder.CX)]);

		renorme(encoder);
	}
	else
		encoder.C = encoder.C + p;
}

__device__ void codelps(MQEncoder &encoder)
{
	unsigned int p = Qe[getI(encoder, encoder.CX)];

	encoder.A = encoder.A - p;

	if(encoder.A < p)
		encoder.C = encoder.C + p;
	else
		encoder.A = p;

	if(SWITCH[getI(encoder, encoder.CX)])
		SwitchNthBit(encoder.CXMPS, encoder.CX);

	setI(encoder, encoder.CX, NLPS[getI(encoder, encoder.CX)]);
	
	renorme(encoder);
}

__device__ void setbits(MQEncoder &encoder)
{
	unsigned int TEMPC = encoder.C + encoder.A;
	encoder.C = encoder.C | 0xFFFF;

	if(encoder.C >= TEMPC)
		encoder.C = encoder.C - 0x8000;
}

__device__ int mqFlush(MQEncoder &encoder)
{
	int nbits = 12 - encoder.CT; // the number of bits we need to flush out of C (27 - 15 - Ct)
	//encoder.C <<= encoder.CT; // move the next 8 available bits into the partial byte
	
	while(nbits > 0)
	{
		encoder.C <<= encoder.CT; // move bits into available positions for next transfered
		encoder.CT = 0;
		byteout(encoder);
		nbits -= encoder.CT; // new value of Ct is the number of bits just transferred
	}
	
	byteout(encoder);
	byteout(encoder);
	
	return encoder.L;
}

/* old version {
	setbits(encoder);
	encoder.C <<= encoder.CT;

	byteout(encoder);

	encoder.C <<= encoder.CT;

	byteout(encoder);

	if(encoder.outbuf[encoder.L] != (unsigned char) 0xFF)
		encoder.L++;

	return encoder.L;
}*/

//#include "gpu_emu_64_arit.cuh"

typedef long long int __int64;

__device__ int calcFmin(MQEncoder &encoder)
{
	__int64 sop = encoder.C;
	sop <<= encoder.CT;
	__int64 Cr = encoder.T;
	Cr <<= 27;
	Cr += sop;
	__int64 Ar = encoder.A;
	Ar <<= encoder.CT;
	__int64 Rf = 0, s = 8;
	unsigned short Sf = 35, F = 0;

	while(F < 5 && (Rf + (((__int64) 1) << Sf) - 1 < Cr || Rf - (1 << Sf) - 1 >= Cr + Ar))
	{
		F++;
		if(F <= 4)
		{
			Sf -= s;
			Rf += ((__int64) encoder.outbuf[encoder.L + F - 1]) << Sf;

			s = 8;
			s -= (encoder.outbuf[encoder.L + F - 1] == 0xFF);
		}
	}

	return F;
}

__device__ int mqFullFlush(MQEncoder &encoder)
{
	int Fmin = calcFmin(encoder);
	
	return Fmin + encoder.L;
}

__device__ void mqEncode(MQEncoder &encoder, int decision, int context)
{
	#ifdef DEBUG_MQ
	/* debug purposes */
	Cstates[l++] = encoder.C;
	Cstates[l++] = decision;
	Cstates[l++] = context;
	/* */
	#endif
	
	encoder.CX = context;
	
	if(decision == GetNthBit(encoder.CXMPS, encoder.CX))
		codemps(encoder);
	else
		codelps(encoder);
}

__device__ void bytein(MQDecoder &decoder)
{
	decoder.CT = 8;
	if(decoder.L == decoder.Lmax - 1 || (decoder.T == (unsigned char) 0xFF && decoder.NT > (unsigned char) 0x8F))
		decoder.C += 0xFF00;
	else
	{
		if(decoder.T == (unsigned char) 0xFF)
			decoder.CT = 7;

		decoder.T = decoder.NT;
		decoder.NT = decoder.outbuf[decoder.L + 1];
		decoder.L++;
		decoder.C += decoder.T << (16 - decoder.CT);
	}
}

__device__ void renormd(MQDecoder &decoder)
{
	do
	{
		if(decoder.CT == 0)
			bytein(decoder);

		decoder.A <<= 1;
		decoder.C <<= 1;
		decoder.CT -= 1;
	}
	while((decoder.A & 0x8000) == 0);
}

__device__ int lps_exchange(MQDecoder &decoder)
{
	unsigned int p = Qe[getI(decoder, decoder.CX)];
	int D;

	if(decoder.A < p)
	{
		decoder.A = p;
		D = GetNthBit(decoder.CXMPS, decoder.CX);
		setI(decoder, decoder.CX, NMPS[getI(decoder, decoder.CX)]);
	}
	else
	{
		decoder.A = p;
		D = 1 - GetNthBit(decoder.CXMPS, decoder.CX);

		if(SWITCH[getI(decoder, decoder.CX)])
			SwitchNthBit(decoder.CXMPS, decoder.CX);

		setI(decoder, decoder.CX, NLPS[getI(decoder, decoder.CX)]);
	}

	return D;
}

__device__ int mps_exchange(MQDecoder &decoder)
{
	unsigned int p = Qe[getI(decoder, decoder.CX)];
	int D;

	if(decoder.A < p)
	{
		D = 1 - GetNthBit(decoder.CXMPS, decoder.CX);

		if(SWITCH[getI(decoder, decoder.CX)])
			SwitchNthBit(decoder.CXMPS, decoder.CX);
		
		setI(decoder, decoder.CX, NLPS[getI(decoder, decoder.CX)]);
	}
	else
	{
		D = GetNthBit(decoder.CXMPS, decoder.CX);
		setI(decoder, decoder.CX, NMPS[getI(decoder, decoder.CX)]);
	}

	return D;
}

__device__ void FillLSBs(MQDecoder &decoder)
{
	decoder.CT = 8;
	if(decoder.L == decoder.Lmax || (decoder.T == (unsigned char) 0xFF && decoder.NT > (unsigned char) 0x8F))
		decoder.C += (unsigned char) 0xFF;
	else
	{
		if(decoder.T == (unsigned char) 0xFF)
			decoder.CT = 7;

		decoder.T = decoder.NT;
		decoder.L++;
		decoder.NT = decoder.outbuf[decoder.L];
		decoder.C += ((unsigned char) decoder.T) << (8 - decoder.CT);
	}
}

__device__ void mqInitDec(MQDecoder &decoder, byte *inbuf, int codeLength)
{
/*
	decoder.T = 0;
	decoder.Lmax = codeLength;
	decoder.outbuf = inbuf;
	decoder.NT = decoder.outbuf[0];
	decoder.L = 0;
	decoder.C = 0;
	
	FillLSBs(decoder);
	decoder.C <<= decoder.CT;
	FillLSBs(decoder);
	decoder.C <<= 7;

	decoder.CT -= 7;
	decoder.A = (unsigned char) 0x8000;
*/
	decoder.outbuf = inbuf;

	decoder.L = -1;
	decoder.Lmax = codeLength;
	decoder.T = 0;
	decoder.NT = 0;

	bytein(decoder);
	bytein(decoder);

	decoder.C = ((unsigned char) decoder.T) << 16;

	bytein(decoder);

	decoder.C <<= 7;
	decoder.CT -= 7;
	decoder.A = 0x8000;
}

__device__ int getActive(MQDecoder &decoder)
{
	return (decoder.C >> 16) & (unsigned int) 0xFFFF;
}

__device__ void setActive(MQDecoder &decoder, int value)
{
	decoder.C = (decoder.C & (unsigned int) 0xFFFF) | ((value & (unsigned int) 0xFFFF) << 16); 
}

__device__ int mqDecode(MQDecoder &decoder, int context)
{
/*
	decoder.CX = context;

	int s = GetNthBit(decoder.CXMPS, decoder.CX);
	unsigned int p = Qe[getI(decoder, decoder.CX)];
	int out;

	decoder.A -= p;
	
	if(decoder.A < p)
		s = 1 - s;
	
	if(getActive(decoder) < p)
	{
		out = 1 - s;
		decoder.A = p;
	}
	else
	{
		out = s;
		setActive(decoder, getActive(decoder) - p);
	}
	
	if(decoder.A < 0x8000)
	{
		if(out == GetNthBit(decoder.CXMPS, decoder.CX))
			setI(decoder, decoder.CX, NMPS[getI(decoder, decoder.CX)]);
		else
		{
			if(SWITCH[getI(decoder, decoder.CX)])
				SwitchNthBit(decoder.CXMPS, decoder.CX);
			
			setI(decoder, decoder.CX, NLPS[getI(decoder, decoder.CX)]);
		}
		
		while(decoder.A < 0x8000)
		{
			if(decoder.CT == 0)
				bytein(decoder);
			
			decoder.A <<= 1;
			decoder.C <<= 1;
			decoder.CT--;
		}
	}
*/

	decoder.CX = context;

	unsigned int p = Qe[getI(decoder, decoder.CX)];
	int out;
	decoder.A -= p;

	if((decoder.C >> 16) < p)
	{
		out = lps_exchange(decoder);
		renormd(decoder);
	}
	else
	{
		// decrement 16 most significant bits of C register by p
		decoder.C = (decoder.C & 0x0000FFFF) | (((decoder.C >> 16) - p) << 16);

		if((decoder.A & 0x8000) == 0)
		{
			out = mps_exchange(decoder);
			renormd(decoder);
		}
		else
		{
			out = GetNthBit(decoder.CXMPS, decoder.CX);
		}
	}

	#ifdef DEBUG_MQ
	/* debug purposes */ 
	Cstates[l++] = 1;
	Cstates[l++] = out;
	Cstates[l++] = context;
	/* */
	#endif

	return out;
}

#endif /* GPU_MQ_CODER_CUH_ */
