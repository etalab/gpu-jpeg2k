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

int main(int argc, char **argv)
{
//	dbg_fwt_1d();
//	dbg_iwt_1d();
//	dbg_fiwt_1d();
//	dbg_wt();
//	dbg_quantization();
	encode_tasks_test(argv[1]);

	return 0;
}
