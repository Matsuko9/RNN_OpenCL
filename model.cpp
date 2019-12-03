#define __CL_ENABLE_EXCEPTIONS 

#include <iostream>
#include "Utils/utils.hpp"
//#include <CL/cl.hpp>
#include <fstream>
#include "xcl2.hpp"
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

int *model(int *encoded_msg, int msg_len, struct embedding *embed, struct rnn *recurr, struct linear *fc)
{
	float *embedding_weights;
	float *embed_results = new float[msg_len * embed->embed_dim];
	float *hidden_results = new float[msg_len * recurr->hidden_size];
	float *linear_results = new float[msg_len * fc->rows];

	float *rnn_ih_weights = new float[recurr->hidden_size * recurr->in_size];
	float *rnn_ih_bias = new float[recurr->hidden_size];
	float *rnn_hh_bias = new float[recurr->hidden_size];
	float *rnn_hh_weights = new float[recurr->hidden_size * recurr->hidden_size];
	float *linear_weights = new float[fc->rows * fc->cols];
	float *linear_bias = new float[fc->rows];
	int *encoded_original = new int[msg_len];

	embedding_weights = readEmbedWeights(embed->vocab_size, embed->embed_dim, embed->file);
	rnn_ih_weights = readEmbedWeights(recurr->hidden_size, recurr->in_size, recurr->ih_file);
	rnn_ih_bias = read_bias_fc(recurr->hidden_size, recurr->ih_bias_file);
	rnn_hh_bias = read_bias_fc(recurr->hidden_size, recurr->hh_bias_file);
	rnn_hh_weights = readEmbedWeights(recurr->hidden_size, recurr->hidden_size, recurr->hh_file);
	linear_weights = readEmbedWeights(fc->rows, fc->cols, fc->weight_file);
	linear_bias = read_bias_fc(fc->rows, fc->bias_file);

	//print_bias_fc(fc->rows, linear_bias);
	//print_embed_weights(fc->rows, fc->cols, linear_weights);
	//print_encoded(encoded_msg, msg_len);
	try {
		/* ---- One Time OpenCL Initializations ---- */
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);

		std::vector<cl::Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

		cl::Context context(devices);

		cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
		
		std::ifstream sourceFile("kernel.cl");
	      std::string sourceCode(
	         std::istreambuf_iterator<char>(sourceFile),
	         (std::istreambuf_iterator<char>()));
	      cl::Program::Sources source(1,
	         std::make_pair(sourceCode.c_str(),
	         sourceCode.length() + 1));

	    
	    cl::Program program = cl::Program(context, source);
	    program.build(devices);

	    cl::Kernel embed_kernel(program, "embed_kernel");
	    cl::Kernel rnn_kernel(program, "rnn_kernel");
	    cl::Kernel linear_kernel(program, "linear_kernel");

		/* ---- One Time OpenCL Initializations ---- */
	    
		/* ---- Put Buffers on Device ---- */
		cl::Buffer embedWeightsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, embed->vocab_size*embed->embed_dim*sizeof(float));
		queue.enqueueWriteBuffer(embedWeightsBuffer, CL_TRUE, 0, embed->vocab_size*embed->embed_dim*sizeof(float), embedding_weights);

		cl::Buffer linearWeightsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, fc->rows*fc->cols*sizeof(float));
		queue.enqueueWriteBuffer(linearWeightsBuffer, CL_TRUE, 0, fc->rows*fc->cols*sizeof(float), linear_weights);

		cl::Buffer embedResultsBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, msg_len*embed->embed_dim*sizeof(float));
		queue.enqueueWriteBuffer(embedResultsBuffer, CL_TRUE, 0, msg_len*embed->embed_dim*sizeof(float), embed_results);

		cl::Buffer msgBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, msg_len*sizeof(int));
		queue.enqueueWriteBuffer(msgBuffer, CL_TRUE, 0, msg_len*sizeof(int), encoded_msg);

		cl::Buffer dimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		queue.enqueueWriteBuffer(dimBuffer, CL_TRUE, 0, sizeof(int), &embed->embed_dim);

		cl::Buffer rnnihBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float)*recurr->hidden_size*recurr->in_size);
		queue.enqueueWriteBuffer(rnnihBuffer, CL_TRUE, 0, recurr->hidden_size*recurr->in_size*sizeof(float), rnn_ih_weights);

		cl::Buffer rnnihBiasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float)*recurr->hidden_size);
		queue.enqueueWriteBuffer(rnnihBiasBuffer, CL_TRUE, 0, recurr->hidden_size*sizeof(float), rnn_ih_bias);

		cl::Buffer rnnhhBiasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float)*recurr->hidden_size);
		queue.enqueueWriteBuffer(rnnhhBiasBuffer, CL_TRUE, 0, recurr->hidden_size*sizeof(float), rnn_hh_bias);

		cl::Buffer linearBiasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float)*fc->rows);
		queue.enqueueWriteBuffer(linearBiasBuffer, CL_TRUE, 0, fc->rows*sizeof(float), linear_bias);

		cl::Buffer rnnhhBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(float)*recurr->hidden_size*recurr->hidden_size);
		queue.enqueueWriteBuffer(rnnhhBuffer, CL_TRUE, 0, recurr->hidden_size*recurr->hidden_size*sizeof(float), rnn_hh_weights);

		queue.finish();
		/* ---- Put Buffers on Device ---- */


		/* ---- Run Embedding Layer Starts---- */
		embed_kernel.setArg(0, embedWeightsBuffer);
		embed_kernel.setArg(1, msgBuffer);
		embed_kernel.setArg(2, embedResultsBuffer);
		embed_kernel.setArg(3, dimBuffer);

		cl::NDRange global(msg_len,1);
		cl::NDRange local(1,1);

		queue.enqueueNDRangeKernel(embed_kernel, cl::NullRange, global, local);
		
		queue.enqueueReadBuffer(embedResultsBuffer, CL_TRUE, 0, msg_len*embed->embed_dim*sizeof(float), embed_results);
		
		queue.finish();

		/* ---- Run Embedding Layer Ends---- */
		float *hidden_state = new float[recurr->hidden_size];
		for(int i=0; i<recurr->hidden_size; i++)
			*(hidden_state + i) = 0.0;

		cl::Buffer rnnInputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, embed->embed_dim*msg_len*sizeof(float));
		queue.enqueueWriteBuffer(rnnInputBuffer, CL_TRUE, 0, embed->embed_dim*msg_len*sizeof(float), embed_results);

		cl::Buffer rnnOutputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, msg_len*sizeof(float)*recurr->hidden_size);
		queue.enqueueWriteBuffer(rnnOutputBuffer, CL_TRUE, 0, msg_len*sizeof(float)*recurr->hidden_size, hidden_results);

		cl::Buffer hsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		queue.enqueueWriteBuffer(hsizeBuffer, CL_TRUE, 0, sizeof(int), &recurr->hidden_size);

		cl::Buffer hiddenStateBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float)*embed->embed_dim);
		queue.enqueueWriteBuffer(hiddenStateBuffer, CL_TRUE, 0, sizeof(float)*embed->embed_dim, hidden_state);

		queue.finish();

		/* ---- RNN Layer Starts ---- */
		for(int i=0; i<msg_len; i++)
		{
			cl::Buffer iterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
			queue.enqueueWriteBuffer(iterBuffer, CL_TRUE, 0, sizeof(int), &i);

			rnn_kernel.setArg(0, rnnihBuffer);
			rnn_kernel.setArg(1, iterBuffer);
			rnn_kernel.setArg(2, rnnInputBuffer);
			rnn_kernel.setArg(3, rnnOutputBuffer);
			rnn_kernel.setArg(4, dimBuffer);
			rnn_kernel.setArg(5, hsizeBuffer);
			rnn_kernel.setArg(6, rnnihBiasBuffer);
			rnn_kernel.setArg(7, rnnhhBuffer);
			rnn_kernel.setArg(8, rnnhhBiasBuffer);
			rnn_kernel.setArg(9, hiddenStateBuffer);

			cl::NDRange global(1, 1);
			cl::NDRange local(1, 1);

			queue.enqueueNDRangeKernel(rnn_kernel, cl::NullRange, global, local);
			queue.enqueueReadBuffer(rnnOutputBuffer, CL_TRUE, 0, msg_len*sizeof(float)*recurr->hidden_size, hidden_results);

			//cl::CommandQueue finish;
			//cl_command_queue queue = finish();
			queue.finish();
		}

		/* ---- RNN Layer Ends ---- */


		/* ---- Linear Layer starts ---- */

		cl::Buffer linearOutputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, msg_len*sizeof(float)*fc->rows);
		queue.enqueueWriteBuffer(linearOutputBuffer, CL_TRUE, 0, msg_len*sizeof(float)*fc->rows, linear_results);

		cl::Buffer vocabsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		queue.enqueueWriteBuffer(vocabsizeBuffer, CL_TRUE, 0, sizeof(int), &fc->rows);

		linear_kernel.setArg(0, linearWeightsBuffer);
		linear_kernel.setArg(1, linearBiasBuffer);
		linear_kernel.setArg(2, rnnOutputBuffer);
		linear_kernel.setArg(3, linearOutputBuffer);
		linear_kernel.setArg(4, vocabsizeBuffer);
		linear_kernel.setArg(5, hsizeBuffer);

		cl::NDRange global_linear(msg_len, 1);
		cl::NDRange local_linear(1, 1);

		queue.enqueueNDRangeKernel(linear_kernel, cl::NullRange, global_linear, local_linear);

		queue.enqueueReadBuffer(linearOutputBuffer, CL_TRUE, 0, msg_len*fc->rows*sizeof(float), linear_results);

		queue.finish();
		/* ---- Linear Layer Ends ---- */
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}

	/*
	for(int i=0;i <msg_len; i++)
	{
		for(int j=0; j<embed->embed_dim; j++)
		{
			std::cout<<*(embed_results + i*embed->embed_dim + j);
		}
		std::cout<<"\n";
	}*/


	for(int i=0; i<msg_len; i++)
		encoded_original[i] = find_max( i*fc->rows, i*fc->rows + fc->rows, linear_results, i);

	//print_encoded(encoded_original, msg_len);

	return encoded_original;
}	
