__kernel void embed_kernel(__constant float *embed_weights, __global int* seq, __global float *results, __constant int* dim)
{
	int work_id = get_global_id(0);

	int word = *(seq + work_id);

	for(int i=0; i<*dim; i++)
	{
		*(results + work_id**dim + i) = *(embed_weights + word**dim + i);
	}
}

__kernel void linear_kernel(__constant float *weights, __constant float *bias, __global float* input, __global float* output, __constant int* rows, __constant int *cols)
{
	int work_id = get_global_id(0);

	for(int i=0; i<*rows; i++)
	{
		float sum = 0.0;
		for(int j=0; j<*cols; j++)
		{
			sum += *(weights + i**cols + j) * *(input + work_id**cols + j);
		}
		*(output + work_id**rows + i) = sum;
		*(output + work_id**rows + i) += *(bias + i);
	}

	/*
	if (work_id == 1){
		for(int i=0;i<*rows; i++)
		{
			printf("%f ", *(output + work_id**rows + i));
		}
	}*/

}

__kernel void rnn_kernel(__constant float *weight_ih, __global int* iter, __constant float *input, __global float* output, __constant int* dim, __constant int* hsize, __constant float* bias_ih, __constant float* weight_hh, __constant float* bias_hh, __global float* hidden_state)
{

	/* w_ih x embed_out */

	for(int i=0; i<*hsize; i++)
	{
		float sum = 0.0;
		for(int j=0; j<*dim; j++)
		{

			sum += *(weight_ih + i **dim + j) * *(input + *iter**dim + j);
		}

		*(output + *iter**hsize + i) = sum;
		*(output + *iter**hsize + i) += *(bias_ih + i);
	}

	/* w_hh x hidden */

	for(int i=0; i<*hsize; i++)
	{
		float sum = 0.0;
		for(int j=0; j<*hsize; j++)
		{

			sum += *(weight_hh + i **hsize + j) * *(hidden_state + j);
		}

		*(output + *iter**hsize + i) += sum;
		*(output + *iter**hsize + i) += *(bias_hh + i);

	}

	/* --- Tanh ---- */

	for(int i=0; i<*hsize; i++)
		*(output + *iter**hsize + i) = tanh(*(output + *iter**hsize + i));
	/* --- Tanh ---- */

	for(int i=0; i<*hsize; i++)
		*(hidden_state + i) = *(output + *iter**hsize + i);

	/*
	if (*iter == 0){
		for(int i=0;i<*hsize; i++)
		{

			printf("%f ", *(output + *iter**hsize + i));
		}
	}*/

}
