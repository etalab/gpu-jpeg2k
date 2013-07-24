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
 * @file tagTreeEncode.h
 *
 * @author milosz 
 * @date 03-11-2010
 */

#ifndef TAGTREEENCODE_H_
#define TAGTREEENCODE_H_

#define QUAD_OFFSET_X 1
#define QUAD_OFFSET_Y 1
#define QUAD_SIZE_X 2
#define QUAD_SIZE_Y 2

#include <stdint.h>

typedef struct type_tag_tree_node type_tag_tree_node;
typedef struct type_tag_tree type_tag_tree;

/**
 * Tag node
 */
struct type_tag_tree_node {
  type_tag_tree_node *parent;
  int value;
  int low;
  int known;
};

/**
 * Tag tree
 */
struct type_tag_tree {
  int num_leafs_h;
  int num_leafs_v;
  int num_nodes;
  type_tag_tree_node *nodes;
};

#endif /* TAGTREEENCODE_H_ */
