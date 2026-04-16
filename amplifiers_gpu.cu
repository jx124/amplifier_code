#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

constexpr float eps = 1e-6;
constexpr size_t alpha_points = 600;
constexpr size_t beta_points = 600;
constexpr size_t gamma_points = 600;
constexpr size_t phi_points = 600;

struct Result {
    float alpha;
    float beta;
    float gamma;
    float phi;
    float value;
};

__host__ __device__ float H(float p) {
    // Returns the entropy of p in base e.
    if (p <= 0 || p >= 1) {
        return 0;
    }
    return - p * std::log(p) - (1 - p) * std::log(1 - p);
}

__host__ __device__ float prob(float alpha, float beta, float gamma, float phi) {
    // Returns 1/r log P'(u, c).
    float common_1 = 6 + beta * phi; // u / (alpha * r)
    float common_2 = gamma * (1 - phi); // c / (alpha * r)
    float result = common_1 * alpha * H(common_2 / common_1) + (12 - common_1 * alpha) * H(common_2 * alpha / (12 - common_1 * alpha))
        + common_2 * alpha / 2 * std::log(common_2 * common_2 * alpha / ((common_1 - common_2) * (12 - common_1 * alpha - common_2 * alpha)))
        + common_1 * alpha / 2 * std::log((common_1 - common_2) * alpha / (12 - common_1 * alpha - common_2 * alpha))
        + 6 * std::log((12 - common_1 * alpha - common_2 * alpha) / 12);
    return result;
}

__host__ __device__ float prob_grid(float alpha, float beta, float gamma, float phi, float d_alpha, float d_beta, float d_gamma, float d_phi) {
    // Returns an overestimate of 1/r log P'(u, c) over the grid cell.
    float result = 0;
    float max_alpha = alpha + d_alpha;
    float max_beta = beta + d_beta;
    float max_gamma = gamma + d_gamma;
    float max_phi = phi + d_phi;

    float min_common_1 = 6 + beta * phi;
    float min_common_2 = gamma * (1 - max_phi);
    float max_common_1 = 6 + max_beta * max_phi;
    float max_common_2 = max_gamma * (1 - phi);

    // common_1 * alpha * H(common_2 / common_1) is always increasing in common_1
    float max_arg_1 = max_common_2 / max_common_1;
    float min_arg_1 = min_common_2 / max_common_1;
    if (max_arg_1 < 0.5f) {
        result += max_common_1 * max_alpha * H(max_arg_1);
    } else if (min_arg_1 > 0.5f) {
        result += max_common_1 * max_alpha * H(min_arg_1);
    } else {
        result += max_common_1 * max_alpha * H(0.5f);
    }

    // (12 - common_1 * alpha) * H(common_2 * alpha / (12 - common_1 * alpha)) is always increasing in (12 - common_1 * alpha)
    float max_arg_2 = max_common_2 * max_alpha / (12 - min_common_1 * alpha);
    float min_arg_2 = min_common_2 * alpha / (12 - min_common_1 * alpha);
    if (max_arg_2 < 0.5f) {
        result += (12 - min_common_1 * alpha) * H(max_arg_2);
    } else if (min_arg_2 > 0.5f) {
        result += (12 - min_common_1 * alpha) * H(min_arg_2);
    } else {
        result += (12 - min_common_1 * alpha) * H(0.5f);
    }

    // The log terms in the probability expression are factored and simplified to obtain
    // common_2 * alpha * std::log(common_2 * alpha)
    // + (common_1 - common_2) * alpha / 2 * std::log((common_1 - common_2) * alpha)
    // + (12 - common_1 * alpha - common_2 * alpha) / 2 * std::log(12 - common_1 * alpha - common_2 * alpha)
    // -6 * std::log(12)

    // common_2 * alpha * std::log(common_2 * alpha)
    float max_arg_3 = max_common_2 * max_alpha;
    float min_arg_3 = min_common_2 * alpha;
    if (max_arg_3 < std::exp(-1)) {
        result += min_arg_3 * std::log(min_arg_3);
    } else if (min_arg_3 > std::exp(-1)) {
        result += max_arg_3 * std::log(max_arg_3);
    } else {
        result += std::max(min_arg_3 * std::log(min_arg_3), max_arg_3 * std::log(max_arg_3));
    }
    
    // (common_1 - common_2) * alpha / 2 * std::log((common_1 - common_2) * alpha)
    float max_arg_4 = (max_common_1 - min_common_2) * max_alpha;
    float min_arg_4 = (min_common_1 - max_common_2) * alpha;
    if (max_arg_4 < std::exp(-1)) {
        result += min_arg_4 / 2 * std::log(min_arg_4);
    } else if (min_arg_4 > std::exp(-1)) {
        result += max_arg_4 / 2 * std::log(max_arg_4);
    } else {
        result += std::max(min_arg_4 / 2 * std::log(min_arg_4), max_arg_4 / 2 * std::log(max_arg_4));
    }
    
    // (12 - common_1 * alpha - common_2 * alpha) / 2 * std::log(12 - common_1 * alpha - common_2 * alpha)
    float max_arg_5 = 12 - min_common_1 * alpha - min_common_2 * alpha;
    float min_arg_5 = 12 - max_common_1 * max_alpha - max_common_2 * max_alpha;
    if (max_arg_5 < std::exp(-1)) {
        result += min_arg_5 / 2 * std::log(min_arg_5);
    } else if (min_arg_5 > std::exp(-1)) {
        result += max_arg_5 / 2 * std::log(max_arg_5);
    } else {
        result += std::max(min_arg_5 / 2 * std::log(min_arg_5), max_arg_5 / 2 * std::log(max_arg_5));
    }

    result -= 6 * std::log(12);

    return result;
}

__host__ __device__ float trivial_count(float alpha, float phi) {
    // Returns 2/r log nCr(6r - f, f).
    float result = (12 - phi * alpha) * H(phi * alpha / (12 - phi * alpha));
    return result;
}

__host__ __device__ float trivial_count_grid(float alpha, float phi, float d_alpha, float d_phi) {
    // Returns an overestimate of 2/r log nCr(6r - f, f) over the grid cell.
    float result = 0;
    float max_phi = phi + d_phi;
    float max_alpha = alpha + d_alpha;
    
    // (12 - phi * alpha) * H(phi * alpha / (12 - phi * alpha)) is always increasing in (12 - phi * alpha)
    float max_arg = max_phi * max_alpha / (12 - phi * alpha);
    float min_arg = phi * alpha / (12 - phi * alpha);
    if (max_arg < 0.5) {
        result += (12 - phi * alpha) * H(max_arg);
    } else if (min_arg > 0.5) {
        result += (12 - phi * alpha) * H(min_arg);
    } else {
        result += (12 - phi * alpha) * H(0.5);
    }
    
    return result;
}

__host__ __device__ float alt_counts(float alpha, float beta, float phi) {
    // Returns 2/r log(nCr(r + f, f) * min((u/2 - 3a + 4f, f), (3a + 4f - u/2, f))).
    float smaller_term = std::min(phi * alpha * (beta + 4) * H(1 / (beta + 4)),
            phi * alpha * (4 - beta) * H(1 / (4 - beta)));
    return (2 + phi * alpha) * H(phi * alpha / (2 + phi * alpha)) + smaller_term;
}

__host__ __device__ float alt_counts_grid(float alpha, float beta, float phi, float d_alpha, float d_beta, float d_phi) {
    // Returns an overestimate of 2/r log(nCr(r + f, f) * min((u/2 - 3a + 4f, f), (3a + 4f - u/2, f)))
    // over the grid cell.
    float result = 0;
    float max_alpha = alpha + d_alpha;
    float max_beta = beta + d_beta;
    float max_phi = phi + d_phi;

    // (2 + phi * alpha) * H(phi * alpha / (2 + phi * alpha)) is always increasing in (2 + phi * alpha)
    float max_arg = max_phi * max_alpha / (2 + max_phi * max_alpha);
    float min_arg = phi * alpha / (2 + max_phi * max_alpha);
    if (max_arg < 0.5) {
        result += (2 + max_phi * max_alpha) * H(max_arg);
    } else if (min_arg > 0.5) {
        result += (2 + max_phi * max_alpha) * H(min_arg);
    } else {
        result += (2 + max_phi * max_alpha) * H(0.5);
    }

    // phi * alpha * (beta + 4) * H(1 / (beta + 4)) is always increasing in (beta + 4)
    float d_lower = max_phi * max_alpha * (max_beta + 4) * H(1 / (max_beta + 4));

    // phi * alpha * (4 - beta) * H(1 / (4 - beta)) is always increasing in (4 - beta)
    float d_upper = max_phi * max_alpha * (4 - beta) * H(1 / (4 - beta));

    result += std::min(d_lower, d_upper);

    return result;
}

__host__ __device__ float counts(float alpha, float beta, float phi) {
    // Returns the minimum of the objective over the 3 different counting methods per wheel
    return std::min(trivial_count(alpha, phi), alt_counts(alpha, beta, phi));
}

__host__ __device__ float counts_grid(float alpha, float beta, float phi,
        float d_alpha, float d_beta, float d_phi) {
    // Returns an overestimate of the minimum of the objective over the 3 different counting methods per wheel over the grid cell.
    return std::min(trivial_count_grid(alpha, phi, d_alpha, d_phi),
            alt_counts_grid(alpha, beta, phi, d_alpha, d_beta, d_phi));
}

constexpr size_t TOTAL_THREADS = alpha_points * beta_points * gamma_points;// * phi_points;
constexpr size_t BLOCK_SIZE = 128;
constexpr size_t BLOCKS = (TOTAL_THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;

__global__ void grid_search(Result* device_values_ptr, Result* device_points_ptr) {
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t t_id = threadIdx.x;

    __shared__ Result results[BLOCK_SIZE];
    __shared__ Result point_results[BLOCK_SIZE];
    results[t_id] = {
        -1,
        -1,
        -1,
        -1,
        -std::numeric_limits<float>::infinity()
    };
    point_results[t_id] = {
        -1,
        -1,
        -1,
        -1,
        -std::numeric_limits<float>::infinity()
    };

    if (id >= TOTAL_THREADS) {
        return;
    }

    if (id % BLOCK_SIZE == 0 && blockIdx.x % 100000 == 0) {
        printf("Block %d / %ld (%.2f%% complete)\n", blockIdx.x, BLOCKS, (float)blockIdx.x / (float)BLOCKS * 100.0f);
    }

    size_t alpha_id = id % alpha_points;
    size_t beta_id = (id / alpha_points) % beta_points;
    size_t gamma_id = (id / (alpha_points * beta_points)) % gamma_points;

    float alpha_step = (1.0f - 0.1f) / (alpha_points - 1);
    float beta_step = (6.0f) / (beta_points - 1);
    float gamma_step = (1.0f - eps - eps) / (gamma_points - 1);
    float phi_step = (1.0f - eps - eps) / (phi_points - 1);

    float alpha = 0.1f + alpha_id * alpha_step;
    float beta = -3 + beta_id * beta_step;
    float gamma = eps + gamma_id * gamma_step;

    float max_value = -std::numeric_limits<float>::infinity();
    Result max_result = {
        -1,
        -1,
        -1,
        -1,
        -std::numeric_limits<float>::infinity()
    };
    float max_point_value = -std::numeric_limits<float>::infinity();
    Result max_point_result = {
        -1,
        -1,
        -1,
        -1,
        -std::numeric_limits<float>::infinity()
    };

    for (int i = 0; i < phi_points; i++) {
        //float phi = eps + i * phi_step;
        float phi = eps + i * phi_step;
        float point_prob = prob(alpha, beta, gamma, phi);
        float point_counts = counts(alpha, beta, phi);
        float value_point = point_prob + point_counts;

        if (value_point > max_point_value) {
            max_point_value = value_point;
            max_point_result = {
                alpha,
                beta,
                gamma,
                phi,
                value_point,
            };
        }

        float grid_prob = prob_grid(alpha, beta, gamma, phi, alpha_step, beta_step, gamma_step, phi_step);
        float grid_counts = counts_grid(alpha, beta, phi, alpha_step, beta_step, phi_step);
        float value_grid = grid_prob + grid_counts;
        if (value_grid > max_value) {
            max_value = value_grid;
            max_result = {
                alpha,
                beta,
                gamma,
                phi,
                value_grid,
            };
        }

        if (point_prob > grid_prob) {
            printf("ERROR: Probability underestimation\nPoint prob: %f\nGrid prob: %f",
                    point_prob, grid_prob);
        }
        if (point_counts > grid_counts) {
            printf("ERROR: Probability underestimation\nPoint counts: %f\nGrid counts: %f\n",
                    point_counts, grid_counts);
        }
    }

    results[t_id] = max_result;
    point_results[t_id] = max_point_result;

    __syncthreads();

    // reduce to 1 max result per block
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t_id < s) {
            if (results[t_id + s].value > results[t_id].value) {
                results[t_id] = results[t_id + s];
            }
            if (point_results[t_id + s].value > point_results[t_id].value) {
                point_results[t_id] = point_results[t_id + s];
            }
        }

        __syncthreads();
    }

    if (t_id == 0) {
        device_values_ptr[blockIdx.x] = results[0];
        device_points_ptr[blockIdx.x] = point_results[0];
    }
}

__host__ void print(const Result& result) {
    std::cout << "Max value: " << result.value <<
        "\n\talpha: " << result.alpha <<
        "\n\tbeta: " << result.beta <<
        "\n\tgamma: " << result.gamma <<
        "\n\tphi: " << result.phi <<
        "\n\tprob: " << prob(result.alpha, result.beta, result.gamma, result.phi) <<
        "\n\tcounts: " << counts(result.alpha, result.beta, result.phi) << std::endl;
}

int main() {
    Result* values_ptr = (Result*)malloc(sizeof(Result) * BLOCKS);
    if (!values_ptr) {
        printf("Failed to allocate values_ptr\n");
    }

    Result* points_ptr = (Result*)malloc(sizeof(Result) * BLOCKS);
    if (!points_ptr) {
        printf("Failed to allocate points_ptr\n");
    }

    Result* device_values_ptr;
    Result* device_points_ptr;

    cudaError_t cuda_err = cudaMalloc((void**)&device_values_ptr, sizeof(Result) * BLOCKS);
    if (cuda_err != cudaSuccess) {
        printf("device_values_ptr cudaMalloc error: \"%s\".\n", cudaGetErrorString(cuda_err));
    }

    cuda_err = cudaMalloc((void**)&device_points_ptr, sizeof(Result) * BLOCKS);
    if (cuda_err != cudaSuccess) {
        printf("device_points_ptr cudaMalloc error: \"%s\".\n", cudaGetErrorString(cuda_err));
    }

    grid_search<<<BLOCKS, BLOCK_SIZE>>>(device_values_ptr, device_points_ptr);

    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        printf("cudaDeviceSynchronize error: \"%s\".\n", cudaGetErrorString(cuda_err));
    }

    // copy results from GPU to CPU
    cuda_err = cudaMemcpy(values_ptr, device_values_ptr, sizeof(Result) * BLOCKS, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        printf("device_values_ptr cudaMemcpy error: \"%s\".\n", cudaGetErrorString(cuda_err));
    }
    cuda_err = cudaMemcpy(points_ptr, device_points_ptr, sizeof(Result) * BLOCKS, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        printf("device_points_ptr cudaMemcpy error: \"%s\".\n", cudaGetErrorString(cuda_err));
    }

    float max_prob_grid = -std::numeric_limits<float>::infinity();
    size_t max_id_grid = 0;
    for (size_t i = 0; i < BLOCKS; i++) {
        if (values_ptr[i].value > max_prob_grid) {
            max_prob_grid = values_ptr[i].value;
            max_id_grid = i;
        }
    }

    float max_prob_point = -std::numeric_limits<float>::infinity();
    size_t max_id_point = 0;
    for (size_t i = 0; i < BLOCKS; i++) {
        if (points_ptr[i].value > max_prob_point) {
            max_prob_point = points_ptr[i].value;
            max_id_point = i;
        }
    }

    printf("Grid search id: %ld\n", max_id_grid);
    print(values_ptr[max_id_grid]);

    printf("Point search id: %ld\n", max_id_point);
    print(points_ptr[max_id_point]);

    free(values_ptr);
    free(points_ptr);
    cudaFree(device_values_ptr);
    cudaFree(device_points_ptr);
}
