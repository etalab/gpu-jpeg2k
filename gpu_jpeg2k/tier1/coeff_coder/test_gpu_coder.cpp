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
 * test_gpu_coder.cpp
 *
 *  Created on: Dec 2, 2011
 *      Author: miloszc
 */

void encode_tile() {
	/*
	int buf[4096];

	cuda_memcpy_dth(tasks[i].coefficients, buf, sizeof(int) * tasks[i].nominalWidth * tasks[i].nominalHeight);

	printf("###############################################\n");

	for(int j = 0; j < tasks[i].width; j++)
	{
		for(int k = 0; k < tasks[i].height; k++)
			printf("%d, ", buf[k * tasks[i].nominalWidth + j]);
		printf("\n");
	}

	for(int j = 0; j < tasks[i].length; j++)
		printf("%02X", ((char *) tasks[i].codeStream)[j] + 128);
	printf("\n");

	*/

	// testing only
	/*
	FILE *encf = fopen("encoderStates.txt", "w");

	int l;

	cudaMemcpyFromSymbol((void *) &l, "l", sizeof(int), 0, cudaMemcpyDeviceToHost);

	int *buff = (int *) my_malloc(sizeof(int) * l);

	cudaMemcpyFromSymbol((void *) buff, "Cstates", sizeof(int) * l, 0, cudaMemcpyDeviceToHost);

	fprintf(encf, "sb: %d\n", tasks[i].subband);
	for(int i = 0; i < l; i += 3)
		//binary_fprintf(encf, buff[i + 0]);
		fprintf(encf, "%d C: %08X\tb: %d\tcx: %d\n", i / 3 + 1, buff[i + 0], buff[i + 1], buff[i + 2]);

	free(buff);

	fclose(encf);
	*/
	/*
	int dbgId = 2;
	for(int i = 0; i < dbgId; i++)
		out_cblks.pop_front();

	while(out_cblks.size() > 1)
		out_cblks.pop_back();
	*/
}

void encode_tile_dbg(type_tile *tile)
{
	println_start(INFO);

//	start_measure();

	std::list<type_codeblock *> cblks;

	type_coding_param *coding_params = tile->parent_img->coding_param;

	int size = 64*64*sizeof(int);
	int *buff = (int*)my_malloc(size);
	int x = 0, y = 0;
	type_codeblock *cblk_;

	for(int i = 0; i < tile->parent_img->num_components; i++)
	{
		type_tile_comp *tile_comp = &(tile->tile_comp[i]);
		for(int j = 0; j < tile_comp->num_rlvls; j++)
		{
			type_res_lvl *res_lvl = &(tile_comp->res_lvls[j]);
			for(int k = res_lvl->num_subbands - 1; k >= 0 ; k--)
			{
				type_subband *sb = &(res_lvl->subbands[k]);
				for(int l = 0; l < sb->num_cblks; l++)
				{
					cblk_ = &(sb->cblks[l]);
					cuda_memcpy_dth(cblk_->data_d, buff, size);

					printf("ulx:%d uly:%d magbits:%d orient:%d con:%f\n", sb->tlx, sb->tly, sb->mag_bits, sb->orient, sb->convert_factor);

					for(y = 0; y < cblk_->height; y++)
					{
						for(x = 0; x < cblk_->width; x++)
						{
							printf("%d, ", buff[x + y * 64]);
						}
					}
					printf("\n");
					cblks.push_back(&(sb->cblks[l]));
				}
			}
		}
		printf("\n");
	}

	EntropyCodingTaskInfo *tasks = (EntropyCodingTaskInfo *) my_malloc(sizeof(EntropyCodingTaskInfo) * cblks.size());

	std::list<type_codeblock *>::iterator ii = cblks.begin();

	int num_tasks = 0;
	for(; ii != cblks.end(); ++ii)
		convert_to_task(tasks[num_tasks++], *(*ii));

	printf("%d\n", num_tasks);

	gpuEncode(tasks, num_tasks, coding_params->target_size);

	ii = cblks.begin();

	for(int i = 0; i < num_tasks; i++, ++ii)
	{
		(*ii)->codestream = tasks[i].codeStream;
		(*ii)->length = tasks[i].length;
		(*ii)->significant_bits = tasks[i].significantBits;
	}

	free(tasks);

//	stop_measure(INFO);

	println_end(INFO);
}

void decode_tile() {
	// testing purposes only
	/*
	int l = 0;
	cudaMemcpyToSymbol("l", &l, sizeof(int), 0, cudaMemcpyHostToDevice);
	*/
	/*
	if(num_tasks == 1)
	{

		cuda_memcpy_dth(tasks[0].coefficients, h_coeffs, sizeof(int) * tasks[0].nominalWidth * tasks[0].nominalHeight);
		cudaMemset((void *) tasks[0].coefficients, 0, sizeof(int) * tasks[0].nominalWidth * tasks[0].nominalHeight);
		cudaFree(tasks[0].coefficients);

		right_coeffs = (*ii)->data_d;
	}
	*/

	// testing purposes only
	/*
	FILE *decf = fopen("decoderStates.txt", "w");

	int l;

	cudaMemcpyFromSymbol((void *) &l, "l", sizeof(int), 0, cudaMemcpyDeviceToHost);

	unsigned int *buff = (unsigned int *) my_malloc(sizeof(int) * l);

	cudaMemcpyFromSymbol((void *) buff, "Cstates", sizeof(int) * l, 0, cudaMemcpyDeviceToHost);

	fprintf(decf, "sb: %d\tw: %d\th: %d\n", tasks[i].subband, tasks[i].width, tasks[i].height);
	for(int i = 0; i < l; i += 3)
		//binary_fprintf(decf, buff[i + 0]);
		fprintf(decf, "%d C: %08X\tb: %d\tcx: %d\n", i / 3 + 1, buff[i + 0], buff[i + 1], buff[i + 2]);

	free(buff);

	fclose(decf);
	*/
	/*
	int coef[10000];
	int coef_t[10000];

	cuda_memcpy_dth(tasks[i].coefficients, coef, sizeof(int) * tasks[i].nominalWidth * tasks[i].nominalHeight);
	cuda_memcpy_dth((*ii)->data_d, coef_t, sizeof(int) * tasks[i].nominalWidth * tasks[i].nominalHeight);

	for(int j = 0; j < tasks[i].height; j++)
		for(int k = 0; k < tasks[i].width; k++)
			if(coef_t[j * tasks[i].nominalWidth + k] != coef[j * tasks[i].nominalWidth + k])
			{
				printf("Error in %dth codeblock at %dx%d coefficient (block size %dx%d).\n", i, k, j, tasks[i].width, tasks[i].height);
				printf("%08X %08X\n", coef_t[j * tasks[i].nominalWidth + k], coef[j * tasks[i].nominalWidth + k]);
			}
	*/

	/*
	int buf[4096];

	cuda_memcpy_dth(tasks[i].coefficients, buf, sizeof(int) * tasks[i].nominalWidth * tasks[i].nominalHeight);

	printf("###############################################\n");

	for(int j = 0; j < tasks[i].width; j++)
	{
		for(int k = 0; k < tasks[i].height; k++)
			printf("%d, ", buf[k * tasks[i].nominalWidth + j]);
		printf("\n");
	}

	for(int j = 0; j < tasks[i].length; j++)
		printf("%02X", ((char *) tasks[i].codeStream)[j] + 128);
	printf("\n");

	*/
}

/*void encode_tile2(type_tile *tile)
{
	#include "sample_data.h"

	int *d_co;

	type_coding_param *coding_params = tile->parent_img->coding_param;

	cudaMalloc((void **) &d_co, sizeof(int) * 4096);
	cuda_memcpy_htd(buf, d_co, sizeof(int) * 4096);

	int skipbp = -1;
	for(int bp = 30; bp >= 0; bp--)
	{
		skipbp++;
		for(int i = 0; i < 4096; i++)
			if(((buf[i] >> bp) & 1) == 1)
			{
//				printf("%d %08x %08x\n", i, buf[i], (buf[i]) >> bp);
				bp = -1; // outer loop exit
				break;
			}
	}


	EntropyCodingTaskInfo task;

	task.coefficients = d_co;
	task.subband = 2;
	task.width = 64;
	task.height = 64;
	task.nominalWidth = 64;
	task.nominalHeight = 64;

	task.magbits = 11;

	CHECK_ERRORS(gpuEncode(&task, 1, coding_params->target_size));

	int l;
	CHECK_ERRORS(cudaMemcpyFromSymbol((void *) &l, "l", sizeof(int), 0, cudaMemcpyDeviceToHost));

//	printf("%d\n", l);

	int *m = (int *) my_malloc(sizeof(int) * l);

	CHECK_ERRORS(cudaMemcpyFromSymbol((void *) m, "Cstates", sizeof(int) * l, 0, cudaMemcpyDeviceToHost));

	for(int i = 0; i < l; i+=3)
	{
		;//printf("%d. c: %d d: %d cx: %d\n", i / 3, m[i], m[i + 1], m[i + 2]);
	}

	free(m);


	printf("\n#######################################\nlen: %d sb: %d skip: %d \n", task.length, task.significantBits, skipbp);

	for(int i = 0; i < task.length; i++)
	{
		printf("%02x", ((char *) task.codeStream)[i] + 128);

		if(i % 20 == 19)
			printf("\n");
	}

	printf("\n");
	fflush(stdout);


	printf("\n###########################################\n");
}*/
/*
#include <string>

#include <boost/algorithm/string.hpp>

std::string get_text(const xmlpp::Node &node)
{
	std::string out = "";

	if(const xmlpp::Element *element = dynamic_cast<const xmlpp::Element *>(&node))
	{
		std::stringstream ss("");

		xmlpp::Node::NodeList ch = element->get_children();
		for(xmlpp::Node::NodeList::iterator iter = ch.begin(); iter != ch.end(); ++iter)
			ss << get_text(**iter);

		out = ss.str();
	}
	else if(const xmlpp::TextNode *tn = dynamic_cast<const xmlpp::TextNode *>(&node))
		out = tn->get_content();
	else if(const xmlpp::ContentNode *cn = dynamic_cast<const xmlpp::ContentNode *>(&node))
		out = cn->get_content();

	boost::trim(out);
	return out;
}

int get_int(const xmlpp::Node &node)
{
	return atoi(get_text(node).c_str());
}

void perform_test(const char *test_file)
{
	std::string test_file_path(test_file);

	xmlpp::DomParser parser;
	parser.set_substitute_entities();
	parser.parse_file(test_file_path.c_str());

	std::list<const xmlpp::Element *> codeblocks;

	const xmlpp::Node *pNode = parser.get_document()->get_root_node();

	xmlpp::Node::NodeList list = pNode->get_children();
	// for every codeblock
	for(xmlpp::Node::NodeList::iterator iter = list.begin(); iter != list.end(); ++iter)
	{
		const xmlpp::Element *element = dynamic_cast<const xmlpp::Element *>(*iter);
		if(element)
			codeblocks.push_back(element);
	}

	int n = codeblocks.size();

	EntropyCodingTaskInfo *infos = new EntropyCodingTaskInfo [n];
	int *lengths = new int [n];
	byte **codeStreams = new byte * [n];

	int u = 0;
	for(std::list<const xmlpp::Element *>::iterator iter = codeblocks.begin(); iter != codeblocks.end(); ++iter, ++u)
	{
		xmlpp::Node::NodeList prop = (*iter)->get_children();
		for(xmlpp::Node::NodeList::const_iterator node = prop.begin(); node != prop.end(); ++node)
		{
			const xmlpp::Element *element = dynamic_cast<const xmlpp::Element *>(*node);

			if(element)
			{
				std::string name = element->get_name();

				if(name == "subband")
				{
					std::string text = get_text(*(*node));

					int subband = -1;

					if(text == "LL/LH")
						subband = 0;
					else if(text == "HL")
						subband = 1;
					else if(text == "HH")
						subband = 2;

					infos[u].subband = subband;
				}
				else if(name == "width")
				{
					infos[u].width = infos[u].nominalWidth = get_int(*(*node));
				}
				else if(name == "height")
				{
					infos[u].height = infos[u].nominalHeight = get_int(*(*node));
				}
				else if(name == "magbits")
				{
					infos[u].magbits = get_int(*(*node));
				}
				else if(name == "data")
				{
					std::string data = get_text(*(*node));

					std::vector<std::string> strs;
					boost::split(strs, data, boost::is_any_of(","));

					int buf[5000];

					for(int i = 0; i < strs.size(); i++)
					{
						boost::trim(strs[i]);
						buf[i] = atoi(strs[i].c_str());
					}

					CHECK_ERRORS(cudaMalloc((void **) &infos[u].coefficients, sizeof(int) * strs.size()));

					cuda_memcpy_htd(buf, infos[u].coefficients, sizeof(int) * strs.size());
				}
				else if(name == "codestream")
				{
					xmlpp::Node::NodeList csp = element->get_children();

					for(xmlpp::Node::NodeList::const_iterator it = csp.begin(); it != csp.end(); ++it)
					{
						const xmlpp::Element *csEl = dynamic_cast<const xmlpp::Element *>(*it);

						if(csEl)
						{
							if(csEl->get_name() == "size")
								lengths[u] = get_int(*csEl);
							else if(csEl->get_name() == "data")
							{
								std::string data = get_text(*csEl);

								std::vector<std::string> strs;

								boost::split(strs, data, boost::is_any_of(","));

								codeStreams[u] = new byte [lengths[u]];

								for(int i = 0; i < lengths[u]; i++)
								{
									boost::trim(strs[i]);
									codeStreams[u][i] = (byte) atoi(strs[i].c_str());
								}


							}
						}
					}
				}
			}
		}
	}

	gpuEncode(infos, n);

	int l = 0;
	CHECK_ERRORS(cudaMemcpyFromSymbol((void *) &l, "l", sizeof(int), 0, cudaMemcpyDeviceToHost));

	int *m = (int *) my_malloc(sizeof(int) * l);

	CHECK_ERRORS(cudaMemcpyFromSymbol((void *) m, "Cstates", sizeof(int) * l, 0, cudaMemcpyDeviceToHost));

	for(int i = 0; i < l; i+=1)
		printf("%d. %d %d %d\n", i, m[i], m[i + 1], m[i + 2]);

	free(m);

	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < lengths[i]; j++)
			if(infos[i].codeStream[j] != codeStreams[i][j])
			{
				std::cout << "Not equal code streams. At " << j << " character. " << i << "th task." << std::endl;
				printf("%d x %d, mb: %d, sb: %d\n", infos[i].width, infos[i].height, infos[i].magbits, infos[i].significantBits);

				std::cout << lengths[i] << "(" << infos[i].length << ")" << std::endl;

				for(int j = 0; j < lengths[i]; j++)
					printf("%02X ", ((char *) infos[i].codeStream)[j] + 128);
				printf("\n");

				for(int j = 0; j < lengths[i]; j++)
					printf("%02X ", ((char *) codeStreams[i])[j] + 128);
				printf("\n");

				throw -1;
			}

		printf("%d done\n", i);
		delete [] codeStreams[i];
	}

	delete [] codeStreams;
	delete [] lengths;
	delete [] infos;
}*/
