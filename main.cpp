#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

//#include <CL/cl.hpp>
#include "xcl2.hpp"
#include <string>
#include <map>
#include <iostream>
#include "Utils/utils.hpp"
#include "model.hpp"

struct embedding 
{
	int embed_dim;
	const char* file;
	int vocab_size;
};

struct rnn
{
	int in_size;
	int hidden_size;
	const char* ih_file;
	const char* ih_bias_file;
	const char* hh_file;
	const char* hh_bias_file;
};

struct linear
{
	int rows;
	int cols;
	const char* weight_file;
	const char* bias_file;
};

int main()
{
	/* ---- Initializations ---- */
	std::string encrypted_string = "IMPWFTUDGIFGKFWDIPKVOTKHYHOSRXCGTJUVPJLORLMXGVGSENHWQQ-WDGRALOKQ";
	static const char* embedding_file = "embed_weights.txt";
	static const char* rnn_ih_file = "ih_weights.txt";
	static const char* rnn_ih_bias_file = "ih_bias.txt";
	static const char* rnn_hh_file = "hh_weights.txt";
	static const char* rnn_hh_bias_file = "hh_bias.txt";
	static const char* linear_weights = "linear_weights.txt";
	static const char* linear_bias = "linear_bias.txt";

	std::map<char,int> vocab_map = build_vocab_map();
	std::map<int, char> reverse_map = build_reverse_map(vocab_map);
	//reverse_print_map(reverse_map);

	int embedding_dim = 5;
	int hidden_size = 10;
	/* ---- Initializations ---- */

	/* ---- Pack Embedding Data ---- */
	struct embedding embed;
	embed.embed_dim = embedding_dim;
	embed.file = embedding_file;
	embed.vocab_size = vocab_map.size();
	/* ---- Pack Embedding Data ---- */

	/* ---- Pack RNN Data ---- */
	struct rnn recurr;
	recurr.in_size = embedding_dim;
	recurr.hidden_size = hidden_size;
	recurr.ih_file = rnn_ih_file;
	recurr.ih_bias_file = rnn_ih_bias_file;
	recurr.hh_file = rnn_hh_file;
	recurr.hh_bias_file = rnn_hh_bias_file;
	/* ---- Pack RNN Data ---- */

	/* ---- Pack Linear Data ---- */
	struct linear fc;
	fc.rows = vocab_map.size();
	fc.cols = hidden_size;
	fc.weight_file = linear_weights;
	fc.bias_file = linear_bias;
	/* ---- Pack Linear Data ---- */
	
	int *encoded_word;
	encoded_word = encoded_string(encrypted_string, vocab_map);

	int *model_out = new int[encrypted_string.length()];
	model_out =	model(encoded_word, encrypted_string.length(), &embed, &recurr, &fc);

	std::string unencrypted_msg = get_message(reverse_map, model_out, encrypted_string.length());

	std::cout<<"UnEncrypyed Message : "<<unencrypted_msg<<std::endl;
	return 0;

}
