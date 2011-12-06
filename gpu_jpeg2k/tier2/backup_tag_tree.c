/**
 * @file backup_tag_tree.c
 *
 * @author Milosz Ciznicki
 */

/*
 * TODO:
 * - reduce memory requirement for tree, one dimension for all levels (t*lvl_length + v)
 */

//typedef struct type_tag_tree type_tag_tree;

///** Tag tree parameters */
//struct type_tag_tree{
//	/** Number of tree levels */
//	uint16_t tree_lvls;
//	/** TODO Array of node values at each tree level */
////	int w_vals[/*TREE_LEVELS*/1][/*VALUES*/1];
//	uint16_t **w_vals;
//	/** TODO Thresholds - To determine if sufficient information has been coded to precisely identify the value of node (w_vals[t][i]) */
////	int w_states[/*TREE_LEVELS*/1][/*THRESHOLDS*/1];
//	uint16_t **w_states;
//	/** The horizontal dimension of the level i */
//	uint16_t *widths;
//	/** The vertical dimension of the level i */
//	uint16_t *heights;
//};

/**
 * @brief Initializes Tag Tree.
 *
 * @param tag_tree
 * @param values
 */
/*void init_tag_tree(type_tag_tree *tag_tree, uint16_t *values)
{
	int i, t;

	for(t = 0; t < tag_tree->tree_lvls; t++)
	{
		for(i = 0; i < tag_tree->widths[t] * tag_tree->heights[t]; i++)
		{
			if(t == 0)
			{
				tag_tree->w_vals[0][i] = values[i];
			}
			tag_tree->w_states[t][i] = 0;
		}
	}
}*/

/**
 * @brief Calculates the number of levels for Tag Tree.
 *
 * @param w
 * @param h
 */
/*int calc_tag_tree_levels(int w, int h)
{
	int tree_lvls = 1;

	 Count number of tree levels
	while (w != 1 || h != 1)
	{
		w = (w + 1) >> 1;
		h = (h + 1) >> 1;
		tree_lvls++;
	}

	return tree_lvls;
}*/

/**
 * @brief Creates Tag Tree.
 *
 * @param buffer
 * @param values
 * @param w
 * @param h
 * @return Constructed Tag Tree.
 */
//type_tag_tree *create_tag_tree(uint16_t *values, int w, int h)
//{
//	uint8_t q_size_x, q_size_y;
//	int i, j, k, l, t;
//	uint16_t tmp_w = w;
//	uint16_t tmp_h = h;
//	uint16_t min_val = UINT16_MAX;
//	uint16_t *child_elems;
//	uint16_t *parent_elems;
//	type_tag_tree *tag_tree = (type_tag_tree *)malloc(sizeof(type_tag_tree));
//
//	assert(w > 0 && h > 0);
//
//	/* Allocate memory */
//	tag_tree->tree_lvls = calc_tag_tree_levels(w, h);
//	tag_tree->w_vals = (uint16_t **)malloc(tag_tree->tree_lvls * sizeof(uint16_t *));
//	tag_tree->w_states = (uint16_t **)malloc(tag_tree->tree_lvls * sizeof(uint16_t *));
//	tag_tree->widths = (uint16_t *)malloc(tag_tree->tree_lvls * sizeof(uint16_t));
//	tag_tree->heights = (uint16_t *)malloc(tag_tree->tree_lvls * sizeof(uint16_t));
//
//	t = 0;
//	tag_tree->widths[0] = tmp_w;
//	tag_tree->heights[0] = tmp_h;
//
//	/* Calculate tree dimensions on different levels */
//	while (tmp_w != 1 || tmp_h != 1)
//	{
//		tmp_w = (tmp_w + 1) >> 1;
//		tmp_h = (tmp_h + 1) >> 1;
//		t++;
//		tag_tree->widths[t] = tmp_w;
//		tag_tree->heights[t] = tmp_h;
//	}
//
//	for(t = 0; t < tag_tree->tree_lvls; t++)
//	{
//		tag_tree->w_vals[t] = (uint16_t *)malloc(tag_tree->widths[t] * tag_tree->heights[t] * sizeof(uint16_t));
//		tag_tree->w_states[t] = (uint16_t *)malloc(tag_tree->widths[t] * tag_tree->heights[t] * sizeof(uint16_t));
//	}
//
//	/* Initialize tag tree with default values */
//	init_tag_tree(tag_tree, values);
//
//	/* Calculate tree nodes using input values */
//	for(t = 0; t < tag_tree->tree_lvls - 1; t++)
//	{
//		tmp_w = tag_tree->widths[t];
//		tmp_h = tag_tree->heights[t];
//
//		for(j = 0; j < tmp_h; j += QUAD_SIZE_Y)
//		{
//			for(i = 0; i < tmp_w; i += QUAD_SIZE_X)
//			{
//				/* Compute quad size */
//				q_size_x = (tmp_w - i < QUAD_SIZE_X) ? tmp_w - i : QUAD_SIZE_X;
//				q_size_y = (tmp_h - j < QUAD_SIZE_Y) ? tmp_h - j : QUAD_SIZE_Y;
//
//				child_elems = tag_tree->w_vals[t] + i + j * tmp_w;
//				parent_elems = tag_tree->w_vals[t + 1] + (i >> 1) + (j >> 1) * tag_tree->widths[t + 1];
//
//				min_val = UINT16_MAX;
//
//				/* Compute min of child quad */
//				for(l = 0; l < q_size_y; l++)
//				{
//					for(k = 0; k < q_size_x; k++)
//					{
//						if(min_val > *child_elems)
//						{
//							min_val = *child_elems;
//						}
//						child_elems++;
//					}
//					/* Skip to beginning of next quad */
//					child_elems += tmp_w - q_size_x;
//				}
//				/* Save result in parent node */
//				*parent_elems = min_val;
//			}
//		}
//	}
//
//	return tag_tree;
//}

/*int get_coords(int x, int y, int width, int size)
{
	if(x >= width || y * width + x >= size)
	{
		return -1;
	}
	return y * width + x;
}

int get_parent_value(type_tag_tree *tt, int x, int y, int level)
{
	return tt->w_vals[level + 1][get_coords(x / 2, y / 2, tt->widths[level+1], tt->widths[level+1]*tt->heights[level+1])];
}

int encode_tag_tree2(type_buffer *buffer, type_tag_tree *tt, int x, int y)
{
	int i;
	int lvl_x, lvl_y, width, size;
	int level;
	int parent_val;

	 Root node has to be coded
	if(tt->w_states[tt->tree_lvls - 1][0] == 0)
	{
		tt->w_states[tt->tree_lvls - 1][0] = 1;
		for(i = 0;i < tt->w_vals[tt->tree_lvls - 1][0];i++)
		{
			write_zero_bit(buffer);
			printf("0");
		}
		write_one_bit(buffer);
		printf("1");
	}

	 All levels except zero level
	for(level = tt->tree_lvls - 2; level > 0;level--)
	{
		lvl_x = x/(2*level);
		lvl_y = y/(2*level);
		width = tt->widths[level];
		size = tt->widths[level] * tt->heights[level];

		if(tt->w_states[level][get_coords(lvl_x, lvl_y, width, size)] == 0)
		{
			parent_val = get_parent_value(tt, lvl_x, lvl_y, level);
			tt->w_states[level][get_coords(lvl_x, lvl_y, width, size)] = 1;

			for(i = 0;i < tt->w_vals[level][get_coords(lvl_x, lvl_y, width, size)] - parent_val;i++)
			{
					write_zero_bit(buffer);
					printf("0");
			}
			write_one_bit(buffer);
			printf("1");
		}
	}
	 Zero level, contains root if tree_lvls - 1 = 0
	if(tt->tree_lvls - 1 > 0){
		tt->w_states[0][get_coords(x, y, tt->widths[0], tt->widths[0]*tt->heights[0])] = 1;
		parent_val = tt->w_vals[1][get_coords(x / 2, y / 2, tt->widths[1], tt->widths[1]*tt->heights[1])];
		for(i = 0;i< tt->w_vals[0][get_coords(x, y, tt->widths[0], tt->widths[0]*tt->heights[0])] - parent_val;i++)
		{
					write_zero_bit(buffer);
					printf("0");
		}
		write_one_bit(buffer);
		printf("1");
	}
	return 0;
}*/

/**
 * @brief Tag Tree Encoding procedure. Leaf nodes are at level t = 0. At subsequent levels in the tree,
 * the node value is defined as the minimum of its descendants. The root of the tree at level T.
 *
 * @param n Horizontal index of the element.
 * @param m Vertical index of the element.
 * @param th Threshold used for encoding.
 */
/*void encode_tag_tree_old(type_buffer *buffer, type_tag_tree *tag_tree, int n, int m, int th)
{
	 Lower bound on the relevant node threshold
	int w_min = 0;
	 The location of the relevant level t ancestor
	int idx;
	 Current level in the tree
	int t;

	for (t = tag_tree->tree_lvls - 1; t >= 0; t--)
	{
		idx = (n >> t) + ((tag_tree->widths[t] + (1 << t) - 1) >> t) * (m >> t);
		if (tag_tree->w_states[t][idx] < w_min)
			tag_tree->w_states[t][idx] = w_min;
		while (tag_tree->w_vals[t][idx] >= tag_tree->w_states[t][idx] && tag_tree->w_states[t][idx] < th)
		{
			tag_tree->w_states[t][idx]++;
			if (tag_tree->w_vals[t][idx] >= tag_tree->w_states[t][idx])
			{
				//emit_bit 0
				write_zero_bit(buffer);
				printf("0");
			} else
			{
				//emit_bit 1
				write_one_bit(buffer);
				printf("1");
			}
		}
		w_min = tag_tree->w_vals[t][idx] < tag_tree->w_states[t][idx] ? tag_tree->w_vals[t][idx] : tag_tree->w_states[t][idx];
	}
}*/
