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
 * @file tag_tree_encode.h
 *
 * @author Milosz Ciznicki
 */

#ifndef TAG_TREE_ENCODE_H_
#define TAG_TREE_ENCODE_H_

#include "tag_tree.h"
#include "../types/buffered_stream.h"

type_tag_tree *tag_tree_create(int num_leafs_h, int num_leafs_v);
void encode_tag_tree(type_buffer *buffer, type_tag_tree *tree, int leafno, int threshold);
int decode_tag_tree(type_buffer *buffer, type_tag_tree *tree, int leaf_no, int threshold);

#endif /* TAG_TREE_ENCODE_H_ */
