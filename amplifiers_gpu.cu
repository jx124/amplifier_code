#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

constexpr float eps = 1e-6;
constexpr size_t alpha_points = 200;
constexpr size_t beta_points = 200;
constexpr size_t gamma_points = 200;
constexpr size_t phi_points = 200;

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

__host__ __device__ float prob_grid(float alpha, float beta, float d_alpha, float d_beta) {
    // Returns an overestimate of 1/r log P'(u, c) over the grid cell.
    float result = 0;
    
    float max_alpha = alpha + d_alpha;
    float max_beta = beta + d_beta;
    
    // alpha * H(beta)
    if (max_beta < 0.5) {
        result += (max_alpha) * H(max_beta);
    } else if (beta > 0.5) {
        result += (max_alpha) * H(beta);
    } else {
        result += (max_alpha) * H(0.5);
    }
        
    // (12 - alpha) * H(beta * alpha/(12 - alpha))
    float max_arg = max_beta * max_alpha/(12 - max_alpha);
    if (max_arg < 0.5) {
        result += (12 - alpha) * H(max_arg);
    } else if (beta * alpha/(12 - alpha) > 0.5) {
        result += (12 - alpha) * H(beta * alpha/(12 - alpha));
    } else {
        result += (12 - alpha) * H(0.5);
    }
    
    // beta * alpha * log(beta): negative and decreasing in beta since beta < 1/6 < 1/e
    result += beta * alpha * std::log(beta);
    
    // beta * alpha / 2 * log(alpha)
    if (max_alpha < std::exp(-1)) {
        // negative and decreasing in alpha
        result += beta * alpha / 2 * std::log(alpha);
    } else if (alpha > std::exp(-1) && max_alpha < 1) {
        // negative and increasing in alpha
        result += beta * max_alpha / 2 * std::log(max_alpha);
    } else if (max_alpha >= std::exp(-1) && alpha <= std::exp(-1)) {
        // negative and concave in alpha
        result += std::max(beta * alpha / 2 * std::log(alpha), beta * max_alpha / 2 * std::log(max_alpha));
    } else {
        // non-negative and increasing in alpha
        result += max_beta * max_alpha / 2 * std::log(max_alpha);
    }
    
    // -beta * alpha / 2 * log(1 - beta): positive and increasing for beta > 0 
    result += -max_beta * max_alpha / 2 * std::log(1 - max_beta);
    
    // -alpha / 2 * (1 + beta) * log(12 - alpha - beta * alpha)
    // = -y / 2 * log(12 - y), where y = alpha * (1 + beta) in [0, 7]: y negative and decreasing below 7.29
    float y = alpha * (1 + beta);
    result += -y / 2 * std::log(12 - y);
    
    // alpha / 2 * log(alpha)
    if (max_alpha < std::exp(-1)) {
        // negative and decreasing in alpha
        result += alpha / 2 * std::log(alpha);
    } else if (alpha > std::exp(-1) && max_alpha < 1) {
        // negative and increasing in alpha
        result += max_alpha / 2 * std::log(max_alpha);
    } else if (max_alpha >= std::exp(-1) and alpha <= std::exp(-1)) {
        // negative and concave in alpha
        result += std::max(alpha / 2 * std::log(alpha), max_alpha / 2 * std::log(max_alpha));
    } else {
        // non-negative and increasing in alpha
        result += max_alpha / 2 * std::log(max_alpha);
    }
        
    // alpha / 2 * log(1 - beta): negative and decreasing in beta
    result += alpha / 2 * std::log(1 - beta);
    
    // 6 * log((12 - alpha - beta * alpha) / 12)
    result += 6 * std::log((12 - alpha - beta * alpha) / 12);
    
    return result;
}

__host__ __device__ float trivial_count(float alpha, float phi) {
    // Returns 1/r log nCr(6r - 2f_i, 2f_i).
    float result = (6 - phi * alpha / 2) * H(phi * alpha / (12 - phi * alpha));
    return result;
}

__host__ __device__ float trivial_count_grid(float phi, float xi, float d_phi, float d_xi) {
    // Returns an overestimate of 1/r log nCr(6r - 2f_i, 2f_i) over the grid cell.
    float result = 0;
    
    // (6 - xi * phi) * H(xi * phi / (6 - xi * phi))
    float max_arg = (xi + d_xi) * (phi + d_phi) / (6 - (xi + d_xi) * (phi + d_phi));
    if (max_arg < 0.5) {
        result = (6 - xi * phi) * H(max_arg);
    } else if (xi * phi / (6 - xi * phi) > 0.5) {
        result = (6 - xi * phi) * H(xi * phi / (6 - xi * phi));
    } else {
        result = (6 - xi * phi) * H(0.5);
    }
    
    return result;
}

__host__ __device__ float small_count(float alpha, float beta, float phi){
    // Returns 1/r log(nCr(6r, f_i) * nCr(u_i - f_i, f_i)).
    float result = 6 * H(phi * alpha / 12) + alpha * (6 - beta * phi - phi) / 2 * H(phi / (6 + beta * phi - phi));
    return result;
}

__host__ __device__ float small_count_grid(float alpha, float chi, float phi, float xi, float d_alpha, float d_chi, float d_phi, float d_xi) {
    // Returns an overestimate of 1/r log(nCr(6r, f_i) * nCr(u_i - f_i, f_i)) over the grid cell.
    float result = 0;
    float max_alpha = alpha + d_alpha;
    float max_chi = chi + d_chi;
    float max_phi = phi + d_phi;
    float max_xi = xi + d_xi;
    
    // 6 * H(xi * phi / 12): xi * phi / 12 < 1/12 so always increasing
    result += 6 * H(max_xi * max_phi / 12);
        
    // (chi * alpha - xi * phi / 2) * H(xi * phi / (2 * chi * alpha - xi * phi))
    float max_arg = max_xi * max_phi / (2 * chi * alpha - max_xi * max_phi);
    if (max_arg < 0.5) {
        result += (max_chi * max_alpha - xi * phi / 2) * H(max_arg);
    } else if (xi * phi / (2 * chi * alpha - xi * phi) > 0.5) {
        result += (max_chi * max_alpha - xi * phi / 2) * H(xi * phi / (2 * chi * alpha - xi * phi));
    } else {
        result += (max_chi * max_alpha - xi * phi / 2) * H(0.5);
    }
    
    return result;
}

__host__ __device__ float alt_count(float alpha, float beta, float phi) {
    // Returns 1/r log(nCr(r + 2f_i, 2f_1) * (d_i + 2f_i, 2f_i)),
    // where d_i = u_i - 6a_i + 6f_i or d_i = 6a_i + 6f_i - u_i.
    float d = std::min(beta + 3, 3 - beta);
    float result = (1 + phi * alpha / 2) * H(phi * alpha / (2 + phi * alpha)) + phi * alpha / 2 * (1 + d) * H(1 / (1 + d));
    return result;
}

__host__ __device__ float alt_count_grid(float alpha, float chi, float gamma, float psi, float phi, float xi,
        float d_alpha, float d_chi, float d_gamma, float d_psi, float d_phi, float d_xi) {
    // Returns an overestimate of 1/r log(nCr(r + 2f_i, 2f_1) * (d_i + 2f_i, 2f_i)) over the grid cell,
    // where d_i = u_i - 6a_i + 6f_i or d_i = 6a_i + 6f_i - u_iprobability: <object type:fl.
    float max_alpha = alpha + d_alpha;
    float max_chi = chi + d_chi;
    float max_gamma = gamma + d_gamma;
    float max_psi = psi + d_psi;
    float max_phi = phi + d_phi;
    float max_xi = xi + d_xi;
    
    float max_d_1 = max_chi * max_alpha - 6 * (psi * gamma - max_xi * max_phi / 2);
    float max_d_2 = 6 * (max_psi * max_gamma + max_xi * max_phi / 2) - chi * alpha;
    float max_d = std::min(max_d_1, max_d_2);

    // (1 + xi * phi) * H(xi * phi / (1 + xi * phi)), increasing in xi * phi
    float result = (1 + max_xi * max_phi) * H(max_xi * max_phi / (1 + max_xi * max_phi));

    // (d + xi * phi) * H(xi * phi / (d + xi * phi)), increasing in xi * phi and d
    result += (max_d + max_xi * max_phi) * H(max_xi * max_phi / (max_d + max_xi * max_phi));
    return result;
}

__host__ __device__ float counts(float alpha, float beta, float phi) {
    // Returns the minimum of the objective over the 3 different counting methods per wheel
    float N_1 = std::min(trivial_count(alpha, phi), alt_count(alpha, beta, phi));
    N_1 = std::min(N_1, small_count(alpha, beta, phi));

    return 2 * N_1;
}

__host__ __device__ float counts_grid(float alpha, float beta, float phi, float xi, float chi, float gamma, float psi,
        float d_alpha, float d_beta, float d_phi, float d_xi, float d_chi, float d_gamma, float d_psi) {
    // Returns an overestimate of the minimum of the objective over the 3 different counting methods per wheel over the grid cell.
    float N_1 = std::min(trivial_count_grid(phi, xi, d_phi, d_xi), alt_count_grid(alpha, chi, gamma, psi, phi, xi, d_alpha, d_chi, d_gamma, d_psi, d_phi, d_xi));
    float N_2 = std::min(trivial_count_grid(phi, 1 - xi, d_phi, d_xi), alt_count_grid(alpha, 1 - chi, gamma, 1 - psi, phi, 1 - xi, d_alpha, d_chi, d_gamma, d_psi, d_phi, d_xi));
    
    N_1 = std::min(N_1, small_count_grid(alpha, chi, phi, xi, d_alpha, d_chi, d_phi, d_xi));
    N_2 = std::min(N_2, small_count_grid(alpha, 1 - chi, phi, 1 - xi, d_alpha, d_chi, d_phi, d_xi));

    return N_1 + N_2;
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
        float value_point = prob(alpha, beta, gamma, phi) + counts(alpha, beta, phi);

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
        //float value_grid = prob_grid(alpha, beta, alpha_step, beta_step)
        //    + counts_grid(alpha, beta, phi, xi, chi, gamma, psi, alpha_step, beta_step, phi_step, 0.0f, 0.0f, gamma_step, 0.0f);
        //if (value_grid > max_value) {
        //    max_value = value_grid;
        //    max_result = {
        //        alpha,
        //        beta,
        //        gamma,
        //        phi,
        //        value_grid,
        //    };
        //}
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

//void print_diff(const Result& result) {
//    float alpha = result.alpha;
//    float beta = result.beta;
//    float gamma = result.gamma;
//    float phi = result.phi;
//
//    float prob_val = prob(alpha, beta);
//    float prob_val_mono = prob_grid(alpha, beta, 1.0 / alpha_points, 1.0 / beta_points);
//    std::cout << "probability: " << prob_val << ", grid: " << prob_val_mono << ", " << (prob_val <= prob_val_mono ? "" : "overestimate") << ", diff: " << prob_val_mono - prob_val << std::endl;
//
//    float val = trivial_count(phi, xi);
//    float val_mono = trivial_count_grid(phi, xi, 1.0 / phi_points, 0);
//    std::cout << "trivial count: " << val << ", grid: " << val_mono << ", " << (val <= val_mono ? "" : "overestimate") << ", diff: " << val_mono - val << std::endl;
//
//    val = small_count(alpha, chi, phi, xi);
//    val_mono = small_count_grid(alpha, chi, phi, xi, 1.0 / alpha_points, 0, 1.0 / phi_points, 0);
//    std::cout << "small count: " << val << ", grid: " << val_mono << ", " << (val <= val_mono ? "" : "overestimate") << ", diff: " << val_mono - val << std::endl;
//
//    val = alt_count(alpha, chi, gamma, psi, phi, xi);
//    val_mono = alt_count_grid(alpha, chi, gamma, psi, phi, xi, 1.0 / alpha_points, 0, 1.0 / gamma_points, 0, 1.0 / phi_points, 0);
//    std::cout << "alt count: " << val << ", grid: " << val_mono << ", " << (val <= val_mono ? "" : "overestimate") << ", diff: " << val_mono - val << std::endl;
//
//    val = counts(alpha, beta, phi, xi, chi, gamma, psi);
//    val_mono = counts_grid(alpha, beta, phi, xi, chi, gamma, psi, 1.0 / alpha_points, 1.0 / beta_points, 1.0 / phi_points, 0, 0, 1.0 / gamma_points, 0);
//    std::cout << "counts: " << val << ", monotone: " << val_mono << ", " << (val <= val_mono ? "" : "overestimate") << ", diff: " << val_mono - val << std::endl;
//
//    std::cout << "total: " << prob_val + val << ", monotone: " << prob_val_mono + val_mono << ", " << (prob_val + val <= prob_val_mono + val_mono ? "" : "overestimate")
//        << ", diff: " << prob_val_mono + val_mono - prob_val - val << std::endl;
//}

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
    //print_diff(values_ptr[max_id_grid]);

    printf("Point search id: %ld\n", max_id_point);
    print(points_ptr[max_id_point]);
    //print_diff(points_ptr[max_id_point]);

    free(values_ptr);
    free(points_ptr);
    cudaFree(device_values_ptr);
    cudaFree(device_points_ptr);
}
