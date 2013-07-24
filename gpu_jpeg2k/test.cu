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
 * @file test.c
 *
 * @author Milosz Ciznicki
 */
extern"C" {
//	#include "dwt/dbg_wt.h"
//	#include "misc/dbg_image.h"
//#include "dwt/dbg_fiwt_1d.h"
//#include "dwt/dbg_fwt_1d.h"
//#include "dwt/dbg_iwt_1d.h"
}

#include "tier1/ebcot/test_gpu_coeff_coder.h"
#include "tier1/bpc/test_gpu_bpc.h"

int main(int argc, char **argv)
{
//	dbg_fwt_1d();
//	dbg_iwt_1d();
//	dbg_fiwt_1d();
//	dbg_wt();
//	dbg_quantization();
//	encode_tasks_test(argv[1]);
	encode_bpc_test(argv[1]);

	return 0;
}
