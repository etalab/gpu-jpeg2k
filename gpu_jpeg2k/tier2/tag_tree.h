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
