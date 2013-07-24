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
 * @file tagTreeDecode.c
 *
 * @author milosz 
 * @date 03-11-2010
 */

#include "tag_tree.h"

/**
 * @brief Initialize Tag Tree.
 *
 * @param _w
 * @param _h
 */
void tag_tree_init(int _w, int _h)
{
	w = _w;
	h = _h;
	treeLvls = 1;

	/* Count number of tree levels */
	while (w != 1 || h != 1)
	{
		w = (w + 1) >> 1;
		h = (h + 1) >> 1;
		treeLvls++;
	}
}

/** Page 513
 * @brief Tag tree decoding procedure.
 *
 * @param n Horizontal index of the element.
 * @param m Vertical index of the element.
 * @param th Threshold used for encoding.
 */
void decode_tag_tree(int n, int m, int th)
{
	/* Lower bound on the relevant node threshold */
	int w_min = 0;
	/* The location of the relevant level t ancestor */
	int idx;
	/* Current level in the tree */
	int t;

	for (t = treeLvls; t >= 0; t--)
	{
		idx = (n >> t) * (m >> t);
		if (w_states[t][idx] < w_min)
		{
			w_states[t][idx] = w_min;
			w_vals[t][idx] = w_min;
		}

		while (w_vals[t][idx] == w_states[t][idx] && w_states[t][idx] < th)
		{
			w_states[t][idx]++;
			if (/*read_bit() == 0*/0)
			{
				w_vals[t][idx]++;
			}
		}
		w_min = w_vals[t][idx] < w_states[t][idx] ? w_vals[t][idx] : w_states[t][idx];
	}
}
