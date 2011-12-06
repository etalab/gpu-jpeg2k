/**
 * @file read_file.c
 *
 * @author Milosz Ciznicki
 */

#include <stdio.h>
#include <stdlib.h>

int **get_execution_times(char *file_name) {
	char line[64];
	FILE *f = NULL;
	int cpu_time;
	int gpu_time;
	int ntasks;
	int **e = NULL;

	f = fopen(file_name, "rt");

	if(f == NULL)
	{
		printf("Could not find input file!\n");
		return NULL;
	}

	if(fgets(line, 64, f) != NULL)
	{
		sscanf(line, "%d", &ntasks);
	}

	printf("%d\n", ntasks);

	e = (int **) malloc(ntasks * sizeof(int *));

	if(e == NULL)
	{
		printf("Out of memory\n");
		return NULL;
	}

	int i = 0;
	for(i = 0; i < ntasks; ++i)
	{
		e[i] = (int *) malloc(3 * sizeof(int));
		if(e[i] == NULL)
		{
			printf("Out of memory\n");
			return NULL;
		}
	}

	i = 0;

	while (fgets(line, sizeof(line), f) != NULL)
	{
		sscanf(line, "%d %d %d", &e[i][0], &e[i][1], &e[i][2]);
		//printf("%d %d %d %d\n", i, e[i][0], e[i][1], e[i][2]);
		i++;
	}
	fclose(f);

	return e;
}
