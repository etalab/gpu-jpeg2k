/**
 * @file backup.c
 *
 * @author Milosz Ciznicki
 */

/*int get_binary_length(int n)
{
	int length = 0;

	while(n > 0)
	{
		n >>= 1;
		length++;
	}

	return length;
}

void encode_cblk_length(type_buffer *buffer, int length, int num_coding_passes)
{
	int i;
	int binary_length = get_binary_length(length);
	int lblock = binary_length - (int)floor((float)log(num_coding_passes) / (float)log(2)) - 3;

	if(lblock > 0)
	{
		for(i = 0; i < lblock; i++)
		{
			write_one_bit(buffer);
		}
	}

	write_zero_bit(buffer);

	if(lblock < 0)
	{
		while(lblock < 0)
		{
			write_zero_bit(buffer);
			lblock++;
		}
	}

	write_bits(buffer, length, binary_length);
}

int calcBinaryLength(int bytes){
	int i = 0;
	while(bytes > 0){
		bytes /= 2;
		i++;
	}
	return i;
}

void encode_cblk_length2(type_buffer *buffer, int bytes, int nCodingpasses)
{
	int i, length, temp;
	length = calcBinaryLength(bytes);
	temp = length - (int)floor(log((double)nCodingpasses)/log((double)2));
	temp -= 3;
	printf("inc:%d\n", temp);
	if(temp > 0){
		for(i = 0;i < temp;i++){
			write_one_bit(buffer);
			printf("1");
		}
	}
	write_zero_bit(buffer);
	if(temp < 0){
		while(temp < 0){
			write_zero_bit(buffer);
			printf("0");
			temp++;
		}
	}
	for(i = length - 1; i>=0;i--){
		if(bytes >= power(2,i)){
			bytes -= power(2,i);
			write_one_bit(buffer);
			printf("1");
		}
		else{
			write_zero_bit(buffer);
			printf("0");
		}
	}
}

void print_tag_tree(type_tag_tree *tt)
{
	int i, k, l;
	printf("tree_lvls:%d\n", tt->tree_lvls);

	for(i = 0; i < tt->tree_lvls; i++)
	{
		printf("lvl:%d\n", i);
		for(k = 0; k < tt->heights[i]; k++)
		{
//			printf("first loop\n");
			for(l = 0; l < tt->widths[i]; l++)
			{
//				printf("second loop\n");
				printf("(%d,%d),", tt->w_states[i][l + k * tt->heights[i]], tt->w_vals[i][l + k * tt->heights[i]]);
			}
			printf("\n");
		}
	}
}

int power(int base, int exp){
	int temp,i;
	temp = base;
	if(exp == 0){
		return 1;
	}
	if(exp == 1){
		return base;
	}
	for(i = 1;i < exp;i++){
		temp *= base;
	}
	return temp;
}

int encode_num_coding_passes2(type_buffer *buffer, int n){
	assert(n > 0 && n < 165);
	if(n == 1){
		write_zero_bit(buffer);
		return 0;
	}
	if(n == 2){
		write_one_bit(buffer);
		write_zero_bit(buffer);
		return 0;
	}
	if(n > 2 && n < 6){
		int temp, i;
		write_one_bit(buffer);
		write_one_bit(buffer);
		temp = n - 3;
		for(i = 1;i>-1;i--){
			if(temp >= power(2,i)){
				write_one_bit(buffer);
				temp -= power(2, i);
			}
			else
				write_zero_bit(buffer);
		}
		return 0;
	}
	if(n > 5 && n < 37){
		int i, temp;
		for(i = 0;i<4;i++){
			write_one_bit(buffer);
		}
		temp = n - 6;
		for(i = 4;i>-1;i--){
			if(temp >= power(2, i)){
				write_one_bit(buffer);
				temp -= power(2, i);
			}
			else
				write_zero_bit(buffer);
		}
		return 0;
	}
	if(n > 36){
		int i, temp;
		for(i = 0;i< 8;i++){
			write_one_bit(buffer);
		}
		temp = n - 37;
		for(i = 6;i>-1;i--){
			if(temp >= power(2, i)){
				write_one_bit(buffer);
				temp -= power(2, i);
			}
			else
				write_zero_bit(buffer);
		}
		return 0;
	}
	return 0;
}*/
