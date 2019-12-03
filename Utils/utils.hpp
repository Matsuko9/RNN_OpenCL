#include <string>
#include <map>

std::map<char, int> build_vocab_map();

void print_map(std::map<char,  int>);

int *encoded_string(std::string, std::map<char, int>);

void print_encoded(int *, int);
float *readEmbedWeights(int, int, const char*);
void print_embed_weights(int, int, float*);
float *read_bias_fc(int rows, const char* weight_file);
void *print_bias_fc(int rows, float* bias);
int find_max(int, int, float*, int);
std::map<int, char> build_reverse_map(std::map<char, int>);
void reverse_print_map(std::map<int, char>);
std::string get_message(std::map<int, char>, int*, int);