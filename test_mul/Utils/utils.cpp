#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

std::map<char, int> build_vocab_map()
{
	std::map<char, int> vocab_map;
//	vocab_map.insert(std::make_pair((char)'@', 0));

	for(int i=0; i<26; i++)
	{
		vocab_map.insert(std::make_pair((char)(i + 65), i));
	}

//	vocab_map.insert(std::make_pair((char)'-', 26));

	return vocab_map;
}

void print_map(std::map<char, int> vocab_map)
{	
	std::map<char, int>::iterator it = vocab_map.begin();

	while(it != vocab_map.end())
	{
		std::cout<<it->first<<"->"<<it->second<<std::endl;
		it++;
	}
}
void reverse_print_map(std::map<int, char> vocab_map)
{	
	std::map<int, char>::iterator it = vocab_map.begin();

	while(it != vocab_map.end())
	{
		std::cout<<it->first<<"->"<<it->second<<std::endl;
		it++;
	}
}

int *encoded_string(std::string encrypted_string, std::map<char, int> vocab)
{
	int *encoded_msg = new int[encrypted_string.length()];

	for(int i=0; i<encrypted_string.length(); i++)
	{
		encoded_msg[i] = vocab[(char) encrypted_string[i]];
	}

	return encoded_msg;
}

void print_encoded(int *encoded, int len)
{
	std::cout<<"\n[";
	for(int i=0; i<len; i++)
		std::cout<<*(encoded+i)<<", ";
	std::cout<<"]"<<std::endl;
}

float *readEmbedWeights(int vocab_size, int embed_size, const char* file)
{
	float *weights;

	weights = new float [vocab_size * embed_size];

	FILE *fp = fopen(file, "r");

	if(fp == NULL)
	{
		std::cout<<"Embedding file : "<<file<<" read failed"<<std::endl;
		exit(-1);
	}

	for(int i=0; i<vocab_size; i++)
	{
		for(int j=0; j<embed_size; j++)
		{
			fscanf(fp, "%f\n", weights + i*embed_size + j);
		}
	}

	return weights;
}

void print_embed_weights(int vocab_size, int embed_size, float* weights)
{
	std::cout<<"# ----- Print Embedding Weights Starts ----- #"<<std::endl;

	for(int i=0; i<vocab_size; i++)
	{
		std::cout<<"Row : "<<i+1<<std::endl;
		for(int j=0; j<embed_size; j++)
		{
			std::cout<<*(weights + i*embed_size + j)<<",";
		}
		std::cout<<std::endl;
	}	
	std::cout<<"# ----- Print Embedding Weights Ends ----- #"<<std::endl;	
}

float* read_bias_fc(int rows, const char* bias_file)
{
	float* bias;
	bias = new float [rows];

	FILE *fp = fopen(bias_file, "r");

	if(fp == NULL)
	{
		std::cout<<"Fully connected : bias file read failed!";
		exit(-1);
	}

	for(int i=0; i<rows; i++)
	{
		fscanf(fp, "%f\n", bias + i);
	}

	return bias;
}

void print_bias_fc(int rows, float* bias)
{

	for(int i=0; i<rows; i++)
	{
		std::cout<<"Feature : "<<i+1<<" , Bias : "<<*(bias + i)<<"\n";
	}
}

int find_max(int low, int high, float *array, int iter)
{
	return std::distance(array, std::max_element(array + low, array + high)) - (iter*(high - low));
}

std::map<int, char> build_reverse_map(std::map<char, int> vocab)
{
	std::map<int, char> reverse_map;

	std::map<char, int>::iterator it = vocab.begin();
	
	while(it != vocab.end())
	{
		reverse_map.insert(std::make_pair((int) it->second, (char) it->first));
		it++;
	}
	return reverse_map;
}

std::string get_message(std::map<int, char> reverse_map, int* encoded_word, int msg_len)
{

	std::string tmp;

	for(int i=0; i<msg_len; i++)
		tmp += reverse_map[encoded_word[i]];

	return tmp;

}
