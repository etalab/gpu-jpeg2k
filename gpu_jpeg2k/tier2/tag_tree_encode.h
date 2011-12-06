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
