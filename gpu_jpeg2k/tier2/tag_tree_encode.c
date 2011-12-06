/**
 * @file tag_tree_encode.c
 *
 * @author milosz 
 * @date 25-10-2010
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tag_tree_encode.h"
#include "../print_info/print_info.h"

void tag_tree_reset(type_tag_tree *tree)
{
	int i;

	if (NULL == tree) {
		printf("Error!");
		return;
	}

	for (i = 0; i < tree->num_nodes; i++) {
		tree->nodes[i].value = 999;
		tree->nodes[i].low = 0;
		tree->nodes[i].known = 0;
	}
}

type_tag_tree *tag_tree_create(int num_leafs_h, int num_leafs_v)
{
	int nplh[32];
	int nplv[32];
	type_tag_tree_node *node = NULL;
	type_tag_tree_node *parent_node = NULL;
	type_tag_tree_node *parent_node0 = NULL;
	type_tag_tree *tree = NULL;
	int i, j, k;
	int num_lvls;
	int n;

	tree = (type_tag_tree *) malloc(sizeof(type_tag_tree));
	if (!tree) {
		printf("Error!\n");
		return NULL;
	}
	tree->num_leafs_h = num_leafs_h;
	tree->num_leafs_v = num_leafs_v;

	num_lvls = 0;
	nplh[0] = num_leafs_h;
	nplv[0] = num_leafs_v;
	tree->num_nodes = 0;
	do {
		n = nplh[num_lvls] * nplv[num_lvls];
		nplh[num_lvls + 1] = (nplh[num_lvls] + 1) / 2;
		nplv[num_lvls + 1] = (nplv[num_lvls] + 1) / 2;
		tree->num_nodes += n;
		++num_lvls;
	} while (n > 1);

	/* ADD */
	if (tree->num_nodes == 0) {
		free(tree);
		printf("Error!\n");
		return NULL;
	}

	tree->nodes = (type_tag_tree_node*) calloc(tree->num_nodes, sizeof(type_tag_tree_node));
	if (!tree->nodes) {
		free(tree);
		printf("Error!\n");
		return NULL;
	}

	node = tree->nodes;
	parent_node = &tree->nodes[tree->num_leafs_h * tree->num_leafs_v];
	parent_node0 = parent_node;

	for (i = 0; i < num_lvls - 1; ++i) {
		for (j = 0; j < nplv[i]; ++j) {
			k = nplh[i];
			while (--k >= 0) {
				node->parent = parent_node;
				++node;
				if (--k >= 0) {
					node->parent = parent_node;
					++node;
				}
				++parent_node;
			}
			if ((j & 1) || j == nplv[i] - 1) {
				parent_node0 = parent_node;
			} else {
				parent_node = parent_node0;
				parent_node0 += nplh[i];
			}
		}
	}
	node->parent = 0;

	tag_tree_reset(tree);

	return tree;
}

void tag_tree_setvalue(type_tag_tree *tree, int leaf_no, int value)
{
	type_tag_tree_node *node;
	node = &tree->nodes[leaf_no];
	while (node && node->value > value) {
		node->value = value;
		node = node->parent;
	}
}

void encode_tag_tree(type_buffer *buffer, type_tag_tree *tree, int leaf_no, int threshold)
{
	type_tag_tree_node *stk[31];
	type_tag_tree_node **stkptr;
	type_tag_tree_node *node;
	int low;

	stkptr = stk;
	node = &tree->nodes[leaf_no];
	while (node->parent) {
		*stkptr++ = node;
		node = node->parent;
	}

	low = 0;
	for (;;) {
		if (low > node->low) {
			node->low = low;
		} else {
			low = node->low;
		}

		while (low < threshold) {
			if (low >= node->value) {
				if (!node->known) {
					write_one_bit(buffer);
					//					printf("1");
					node->known = 1;
				}
				break;
			}
			write_zero_bit(buffer);
			//			printf("0");
			++low;
		}

		node->low = low;
		if (stkptr == stk)
			break;
		node = *--stkptr;
	}
}

int decode_tag_tree(type_buffer *buffer, type_tag_tree *tree, int leaf_no, int threshold)
{
	type_tag_tree_node *stk[31];
	type_tag_tree_node **stkptr;
	type_tag_tree_node *node;
	int low;

	stkptr = stk;
	node = &tree->nodes[leaf_no];
	while (node->parent) {
		*stkptr++ = node;
		node = node->parent;
	}

	low = 0;
	for (;;) {
		if (low > node->low) {
			node->low = low;
		} else {
			low = node->low;
		}
		while (low < threshold && low < node->value) {
			if (read_bits(buffer, 1)) {
				node->value = low;
			} else {
				++low;
			}
		}
		node->low = low;
		if (stkptr == stk) {
			break;
		}
		node = *--stkptr;
	}

	return (node->value < threshold) ? 1 : 0;
}
