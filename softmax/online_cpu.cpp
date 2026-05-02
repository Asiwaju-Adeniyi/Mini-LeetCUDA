
#include <cstdlib> 
#include <cstdio> 
#include <cstdint>
#include <vector>
#include <cmath> 

void online_softmax_cpu(float* out, const float* inp, int N, int C) {
    for (int i = 0; i < N; i++) {
        const float* inp_row = inp + i * C;
        float* out_row = out + i * C;

        float maxval = -std::numeric_limits<float>::infinity();
        float sum = 0.0f;

        for (int j = 0; j < C; j++) {
            double maxval_prev = maxval;

            if (inp_row[j] > maxval) {
                maxval = inp_row[j];
                sum = sum * std::exp(maxval_prev - maxval) + std::exp(inp_row[j] - maxval);
            } else {
                sum += std::exp(inp_row[j] - maxval);
            }
        }
        
        for (int j = 0; j < C; j++) {
            out_row[j] = std::exp(inp_row[j] - maxval) / sum;
        }
    }
}


