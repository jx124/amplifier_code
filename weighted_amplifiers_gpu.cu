#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>

constexpr double eps = 1e-8;
constexpr size_t alpha_points = 2001;
constexpr size_t beta_points = 2001;
constexpr size_t gamma_points = 2001;
constexpr size_t phi_points = 2001;

// Enable computation of point values to test if grid values underestimate point values.
constexpr bool ENABLE_TEST = true;

struct Result {
    double alpha;
    double beta;
    double gamma;
    double phi;
    double value;
};

struct Wheel {
    double p_c;
    double p_m;
    double tau;
};

__host__ __device__ double H(double p) {
    // Returns the entropy of p in base e.
    if (p <= 0 || p >= 1) {
        return 0;
    }
    return - p * log(p) - (1 - p) * log(1 - p);
}

__host__ __device__ double prob(double alpha, double beta, double gamma, double phi, Wheel wheel) {
    // Returns 1/r log P'(u, c).
    double p_c = wheel.p_c;
    double p_m = wheel.p_m;
    double tau = wheel.tau;

    double tau_minus_one = tau - 1;
    double two_tau_minus_two = 2 * (tau - 1);

    double common_1 = tau_minus_one + beta * phi / p_c; // u / (alpha * r)
    double common_2 = gamma * (1 - phi) / p_m; // c / (alpha * r)
    double result = common_1 * alpha * H(common_2 / common_1)
        + (two_tau_minus_two - common_1 * alpha) * H(common_2 * alpha / (two_tau_minus_two - common_1 * alpha))
        + common_2 * alpha / 2 * log(common_2 * common_2 * alpha / ((common_1 - common_2) * (two_tau_minus_two - common_1 * alpha - common_2 * alpha)))
        + common_1 * alpha / 2 * log((common_1 - common_2) * alpha / (two_tau_minus_two - common_1 * alpha - common_2 * alpha))
        + tau_minus_one * log((two_tau_minus_two - common_1 * alpha - common_2 * alpha) / two_tau_minus_two);
    return result;
}

__host__ __device__ double prob_grid(double alpha, double beta, double gamma, double phi, double d_alpha, double d_beta, double d_gamma, double d_phi, Wheel wheel) {
    // Returns an overestimate of 1/r log P'(u, c) over the grid cell.
    double p_c = wheel.p_c;
    double p_m = wheel.p_m;
    double tau = wheel.tau;
    double result = 0;
    double max_alpha = alpha + d_alpha;
    double max_beta = beta + d_beta;
    double max_gamma = gamma + d_gamma;
    double max_phi = phi + d_phi;

    double tau_minus_one = tau - 1;
    double two_tau_minus_two = 2 * (tau - 1);

    double min_common_1 = tau_minus_one + beta * phi / p_c;
    double min_common_2 = gamma * (1 - max_phi) / p_m;
    double max_common_1 = tau_minus_one + max_beta * max_phi / p_c;
    double max_common_2 = max_gamma * (1 - phi) / p_m;

    // common_1 * alpha * H(common_2 / common_1) is always increasing in common_1
    double max_arg_1 = max_common_2 / max_common_1;
    double min_arg_1 = min_common_2 / max_common_1;
    if (max_arg_1 < 0.5) {
        result += max_common_1 * max_alpha * H(max_arg_1);
    } else if (min_arg_1 > 0.5) {
        result += max_common_1 * max_alpha * H(min_arg_1);
    } else {
        result += max_common_1 * max_alpha * H(0.5);
    }

    // (two_tau_minus_two - common_1 * alpha) * H(common_2 * alpha / (two_tau_minus_two - common_1 * alpha)) is always increasing in (two_tau_minus_two - common_1 * alpha)
    double max_arg_2 = max_common_2 * max_alpha / (two_tau_minus_two - min_common_1 * alpha);
    double min_arg_2 = min_common_2 * alpha / (two_tau_minus_two - min_common_1 * alpha);
    if (max_arg_2 < 0.5) {
        result += (two_tau_minus_two - min_common_1 * alpha) * H(max_arg_2);
    } else if (min_arg_2 > 0.5) {
        result += (two_tau_minus_two - min_common_1 * alpha) * H(min_arg_2);
    } else {
        result += (two_tau_minus_two - min_common_1 * alpha) * H(0.5);
    }

    // The log terms in the probability expression are factored and simplified to obtain
    // common_2 * alpha * log(common_2 * alpha)
    // + (common_1 - common_2) * alpha / 2 * log((common_1 - common_2) * alpha)
    // + (two_tau_minus_two - common_1 * alpha - common_2 * alpha) / 2 * log(two_tau_minus_two - common_1 * alpha - common_2 * alpha)
    // -tau_minus_one * log(two_tau_minus_two)

    // common_2 * alpha * log(common_2 * alpha)
    double max_arg_3 = max_common_2 * max_alpha;
    double min_arg_3 = min_common_2 * alpha;
    if (max_arg_3 < exp(-1)) {
        result += min_arg_3 * log(min_arg_3);
    } else if (min_arg_3 > exp(-1)) {
        result += max_arg_3 * log(max_arg_3);
    } else {
        result += fmax(min_arg_3 * log(min_arg_3), max_arg_3 * log(max_arg_3));
    }

    // (common_1 - common_2) * alpha / 2 * log((common_1 - common_2) * alpha)
    double max_arg_4 = (max_common_1 - min_common_2) * max_alpha;
    double min_arg_4 = (min_common_1 - max_common_2) * alpha;
    if (max_arg_4 < exp(-1)) {
        result += min_arg_4 / 2 * log(min_arg_4);
    } else if (min_arg_4 > exp(-1)) {
        result += max_arg_4 / 2 * log(max_arg_4);
    } else {
        result += fmax(min_arg_4 / 2 * log(min_arg_4), max_arg_4 / 2 * log(max_arg_4));
    }

    // (two_tau_minus_two - common_1 * alpha - common_2 * alpha) / 2 * log(two_tau_minus_two - common_1 * alpha - common_2 * alpha)
    double max_arg_5 = two_tau_minus_two - min_common_1 * alpha - min_common_2 * alpha;
    double min_arg_5 = two_tau_minus_two - max_common_1 * max_alpha - max_common_2 * max_alpha;
    if (max_arg_5 < exp(-1)) {
        result += min_arg_5 / 2 * log(min_arg_5);
    } else if (min_arg_5 > exp(-1)) {
        result += max_arg_5 / 2 * log(max_arg_5);
    } else {
        result += fmax(min_arg_5 / 2 * log(min_arg_5), max_arg_5 / 2 * log(max_arg_5));
    }

    result -= tau_minus_one * log(two_tau_minus_two);

    return result;
}

__host__ __device__ double trivial_count(double alpha, double phi, Wheel wheel) {
    // Returns 2/r log nCr(6r - f, f).
    double p_c = wheel.p_c;
    double tau = wheel.tau;
    double result = (2 * (tau - 1) - phi * alpha / p_c) * H((phi * alpha / p_c) / (2 * (tau - 1) - phi * alpha / p_c));
    return result;
}

__host__ __device__ double trivial_count_grid(double alpha, double phi, double d_alpha, double d_phi, Wheel wheel) {
    // Returns an overestimate of 2/r log nCr(6r - f, f) over the grid cell.
    double p_c = wheel.p_c;
    double tau = wheel.tau;
    double result = 0;
    double max_phi = phi + d_phi;
    double max_alpha = alpha + d_alpha;
    
    // (2 * (tau - 1) - phi * alpha / p_c) * H((phi * alpha / p_c) / (2 * (tau - 1) - phi * alpha / p_c)) is always increasing in (2 * (tau - 1) - phi * alpha / p_c))
    double max_arg = (max_phi * max_alpha / p_c) / (2 * (tau - 1) - phi * alpha / p_c);
    double min_arg = (phi * alpha / p_c) / (2 * (tau - 1) - phi * alpha / p_c);
    if (max_arg < 0.5) {
        result += (2 * (tau - 1) - phi * alpha / p_c) * H(max_arg);
    } else if (min_arg > 0.5) {
        result += (2 * (tau - 1) - phi * alpha / p_c) * H(min_arg);
    } else {
        result += (2 * (tau - 1) - phi * alpha / p_c) * H(0.5);
    }

    return result;
}

__host__ __device__ double alt_counts(double alpha, double beta, double phi, Wheel wheel) {
    // Returns 2/r log(nCr(r + f, f) * min(nCr(u/2 - (tau - 1)a/2 + (tau + 1)f/2, f), nCr((tau - 1)a/2 + (tau + 1)f/2 - u/2, f))).
    double p_c = wheel.p_c;
    double tau = wheel.tau;
    double smaller_term = fmin(
            (beta + (tau + 1) / 2) * phi * alpha / p_c * H(1 / (beta + (tau + 1) / 2)),
            ((tau + 1) / 2 - beta) * phi * alpha / p_c * H(1 / ((tau + 1) / 2 - beta)));
    return (2 + phi * alpha / p_c) * H((phi * alpha / p_c) / (2 + phi * alpha / p_c)) + smaller_term;
}

__host__ __device__ double alt_counts_grid(double alpha, double beta, double phi, double d_alpha, double d_beta, double d_phi, Wheel wheel) {
    // Returns an overestimate of 2/r log(nCr(r + f, f) * min(nCr(u/2 - (tau - 1)a/2 + (tau + 1)f/2, f), nCr((tau - 1)a/2 + (tau + 1)f/2 - u/2, f))).
    // over the grid cell.
    double p_c = wheel.p_c;
    double tau = wheel.tau;
    double result = 0;
    double max_alpha = alpha + d_alpha;
    double max_beta = beta + d_beta;
    double max_phi = phi + d_phi;

    // (2 + phi * alpha / p_c) * H((phi * alpha / p_c) / (2 + phi * alpha / p_c)) is always increasing in (2 + phi * alpha / p_c)
    double max_arg = (max_phi * max_alpha / p_c) / (2 + max_phi * max_alpha / p_c);
    double min_arg = (phi * alpha / p_c) / (2 + max_phi * max_alpha / p_c);
    if (max_arg < 0.5) {
        result += (2 + max_phi * max_alpha / p_c) * H(max_arg);
    } else if (min_arg > 0.5) {
        result += (2 + max_phi * max_alpha / p_c) * H(min_arg);
    } else {
        result += (2 + max_phi * max_alpha / p_c) * H(0.5);
    }

    // (beta + (tau + 1) / 2) * phi * alpha / p_c * H(1 / (beta + (tau + 1) / 2)) is always increasing in (beta + (tau + 1) / 2)
    double d_lower = (max_beta + (tau + 1) / 2) * max_phi * max_alpha / p_c * H(1 / (max_beta + (tau + 1) / 2));

    // ((tau + 1) / 2 - beta) * phi * alpha / p_c * H(1 / ((tau + 1) / 2 - beta)) is always increasing in ((tau + 1) / 2 - beta)
    double d_upper = ((tau + 1) / 2 - beta) * max_phi * max_alpha / p_c * H(1 / ((tau + 1) / 2 - beta));

    result += fmin(d_lower, d_upper);

    return result;
}

__host__ __device__ double counts(double alpha, double beta, double phi, Wheel wheel) {
    // Returns the minimum of the objective over the 3 different counting methods per wheel
    return fmin(trivial_count(alpha, phi, wheel), alt_counts(alpha, beta, phi, wheel));
}

__host__ __device__ double counts_grid(double alpha, double beta, double phi,
        double d_alpha, double d_beta, double d_phi, Wheel wheel) {
    // Returns an overestimate of the minimum of the objective over the 3 different counting methods per wheel over the grid cell.
    return fmin(trivial_count_grid(alpha, phi, d_alpha, d_phi, wheel),
            alt_counts_grid(alpha, beta, phi, d_alpha, d_beta, d_phi, wheel));
}

__host__ __device__ double tsp_bound(Wheel wheel) {
    double p_c = wheel.p_c;
    double p_m = wheel.p_m;
    double tau = wheel.tau;
    return 3 * (tau - 1) * (4 * p_c + fmax(1.0, p_m)) + 12 * p_c + 20 * fmax(1.0, p_m);
}

__host__ __device__ double atsp_bound(Wheel wheel) {
    double p_c = wheel.p_c;
    double p_m = wheel.p_m;
    double tau = wheel.tau;
    return 3 * (tau - 1) * (2 * p_c + fmax(1.0, fmax(p_c, p_m))) + 6 * p_c + 14 * fmax(1.0, fmax(p_c, p_m));
}

constexpr size_t TOTAL_THREADS = (size_t)alpha_points * beta_points * gamma_points;
constexpr size_t BLOCK_SIZE = 128;
constexpr size_t BLOCKS = (TOTAL_THREADS + BLOCK_SIZE - 1) / BLOCK_SIZE;

__global__ void grid_search(Result* device_values_ptr, Result* device_points_ptr, Wheel wheel) {
    double p_c = wheel.p_c;
    double p_m = wheel.p_m;
    double tau = wheel.tau;
    size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t t_id = threadIdx.x;

    __shared__ Result results[BLOCK_SIZE];
    results[t_id] = {
        -1,
        -1,
        -1,
        -1,
        -std::numeric_limits<double>::infinity()
    };

    __shared__ Result point_results[BLOCK_SIZE];
    if constexpr (ENABLE_TEST) {
        point_results[t_id] = {
            -1,
            -1,
            -1,
            -1,
            -std::numeric_limits<double>::infinity()
        };
    }

    if (id >= TOTAL_THREADS) {
        return;
    }

    if (id % BLOCK_SIZE == 0 && blockIdx.x % 100000 == 0) {
        printf("Block %d / %ld (%.2f%% complete)\n", blockIdx.x, BLOCKS, (double)blockIdx.x / (double)BLOCKS * 100.0);
    }

    size_t alpha_id = id % alpha_points;
    size_t beta_id = (id / alpha_points) % beta_points;
    size_t gamma_id = (id / ((size_t)alpha_points * beta_points)) % gamma_points;

    double alpha_step = (1.0 - 0.1) / (alpha_points - 1);
    double beta_step = (tau - 1.0) / (beta_points - 1);
    double gamma_step = 1.0 / (gamma_points - 1);
    double phi_step = 1.0 / (phi_points - 1);

    double alpha = 0.1 + alpha_id * alpha_step;
    double beta = -(tau - 1.0) / 2.0 + beta_id * beta_step;
    double gamma = gamma_id * gamma_step;

    double max_value = -std::numeric_limits<double>::infinity();
    Result max_result = {
        -1,
        -1,
        -1,
        -1,
        -std::numeric_limits<double>::infinity()
    };

    double max_point_value = -std::numeric_limits<double>::infinity();
    Result max_point_result = {
        -1,
        -1,
        -1,
        -1,
        -std::numeric_limits<double>::infinity()
    };

#pragma unroll
    for (int i = 0; i < phi_points; i++) {
        double phi = i * phi_step;

        double grid_prob = prob_grid(alpha, beta, gamma, phi, alpha_step, beta_step, gamma_step, phi_step, wheel);
        double grid_counts = counts_grid(alpha, beta, phi, alpha_step, beta_step, phi_step, wheel);
        double value_grid = grid_prob + grid_counts;
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

        if constexpr (ENABLE_TEST) {
            double point_prob = prob(alpha, beta, gamma, phi, wheel);
            double point_counts = counts(alpha, beta, phi, wheel);
            double value_point = point_prob + point_counts;

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

            if (point_prob - grid_prob > eps) {
                printf("ERROR: Probability underestimation\nPoint prob: %f\nGrid prob: %f\np_c: %f, p_m: %f, tau: %f\nalpha: %f, beta: %f, gamma: %f, phi: %f\n",
                        point_prob, grid_prob, p_c, p_m, tau, alpha, beta, gamma, phi);
            }
            if (point_counts - grid_counts > eps) {
                printf("ERROR: Probability underestimation\nPoint counts: %f\nGrid counts: %f\np_c: %f, p_m: %f, tau: %f\nalpha: %f, beta: %f, gamma: %f, phi: %f\n",
                        point_counts, grid_counts, p_c, p_m, tau, alpha, beta, gamma, phi);
            }
        }
    }

    results[t_id] = max_result;

    if constexpr (ENABLE_TEST) {
        point_results[t_id] = max_point_result;
    }

    __syncthreads();

    // reduce to 1 max result per block
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t_id < s) {
            if (results[t_id + s].value > results[t_id].value) {
                results[t_id] = results[t_id + s];
            }

            if constexpr (ENABLE_TEST) {
                if (point_results[t_id + s].value > point_results[t_id].value) {
                    point_results[t_id] = point_results[t_id + s];
                }
            }
        }

        __syncthreads();
    }

    if (t_id == 0) {
        device_values_ptr[blockIdx.x] = results[0];

        if constexpr (ENABLE_TEST) {
            device_points_ptr[blockIdx.x] = point_results[0];
        }
    }
}

__host__ void print(const Result& result, Wheel wheel) {
    std::cout << "Max value: " << result.value <<
        "\n\talpha: " << result.alpha <<
        "\n\tbeta: " << result.beta <<
        "\n\tgamma: " << result.gamma <<
        "\n\tphi: " << result.phi <<
        "\n\tprob: " << prob(result.alpha, result.beta, result.gamma, result.phi, wheel) <<
        "\n\tcounts: " << counts(result.alpha, result.beta, result.phi, wheel) << std::endl;
}

int main() {
    // Allocate memory for the grid results on both the CPU and GPU.
    Result* device_values_ptr = nullptr;
    Result* values_ptr = (Result*)malloc(sizeof(Result) * BLOCKS);

    if (!values_ptr) {
        printf("Failed to allocate values_ptr\n");
    }

    cudaError_t cuda_err = cudaMalloc((void**)&device_values_ptr, sizeof(Result) * BLOCKS);
    if (cuda_err != cudaSuccess) {
        printf("device_values_ptr cudaMalloc error: \"%s\".\n", cudaGetErrorString(cuda_err));
    }

    // Allocate memory for the point results on both the CPU and GPU if testing is enabled.
    Result* device_points_ptr = nullptr;
    Result* points_ptr = nullptr;

    if constexpr (ENABLE_TEST) {
        points_ptr = (Result*)malloc(sizeof(Result) * BLOCKS);
        if (!points_ptr) {
            printf("Failed to allocate points_ptr\n");
        }
        cuda_err = cudaMalloc((void**)&device_points_ptr, sizeof(Result) * BLOCKS);
        if (cuda_err != cudaSuccess) {
            printf("device_points_ptr cudaMalloc error: \"%s\".\n", cudaGetErrorString(cuda_err));
        }
    }
    
    double min_tsp_bound = std::numeric_limits<double>::infinity();
    Wheel best_tsp_wheel{};
    double min_atsp_bound = std::numeric_limits<double>::infinity();
    Wheel best_atsp_wheel{};

    double tol = 0.0001;
    double low = 6.6;
    double high = 6.6;
    double mid = (low + high) / 2;
    double prev = 0.0;

    while (low <= high && std::abs(mid - prev) > tol) {
        Wheel wheel = {1.0, 1.0, mid};

        printf("Wheel parameters p_c: %f, p_m: %f, tau: %f\n", wheel.p_c, wheel.p_m, wheel.tau);
        grid_search<<<BLOCKS, BLOCK_SIZE>>>(device_values_ptr, device_points_ptr, wheel);

        cuda_err = cudaDeviceSynchronize();
        if (cuda_err != cudaSuccess) {
            printf("cudaDeviceSynchronize error: \"%s\".\n", cudaGetErrorString(cuda_err));
        }

        // copy results from GPU to CPU
        cuda_err = cudaMemcpy(values_ptr, device_values_ptr, sizeof(Result) * BLOCKS, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            printf("device_values_ptr cudaMemcpy error: \"%s\".\n", cudaGetErrorString(cuda_err));
        }

        if constexpr (ENABLE_TEST) {
            cuda_err = cudaMemcpy(points_ptr, device_points_ptr, sizeof(Result) * BLOCKS, cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                printf("device_points_ptr cudaMemcpy error: \"%s\".\n", cudaGetErrorString(cuda_err));
            }
        }

        double max_prob_grid = -std::numeric_limits<double>::infinity();
        size_t max_id_grid = 0;
        for (size_t i = 0; i < BLOCKS; i++) {
            if (values_ptr[i].value > max_prob_grid) {
                max_prob_grid = values_ptr[i].value;
                max_id_grid = i;
            }
        }

        printf("Grid search id: %ld\n", max_id_grid);
        print(values_ptr[max_id_grid], wheel);

        if constexpr (ENABLE_TEST) {
            double max_prob_point = -std::numeric_limits<double>::infinity();
            size_t max_id_point = 0;
            for (size_t i = 0; i < BLOCKS; i++) {
                if (points_ptr[i].value > max_prob_point) {
                    max_prob_point = points_ptr[i].value;
                    max_id_point = i;
                }
            }

            printf("Point search id: %ld\n", max_id_point);
            print(points_ptr[max_id_point], wheel);
        }

        if (max_prob_grid < 0) {
            high = mid;
        } else {
            low = mid;
        }
        prev = mid;
        mid = (low + high) / 2;

        double tsp = tsp_bound(wheel);
        double atsp = atsp_bound(wheel);

        if (max_prob_grid < 0 && tsp < min_tsp_bound) {
            printf("SUCCESS: TSP bound %.2f/%.2f\n", tsp + 1, tsp);
            min_tsp_bound = tsp;
            best_tsp_wheel = wheel;
        }

        if (max_prob_grid < 0 && atsp < min_atsp_bound) {
            printf("SUCCESS: ATSP bound %.2f/%.2f\n", atsp + 1, atsp);
            min_atsp_bound = atsp;
            best_atsp_wheel = wheel;
        }
        printf("Best TSP wheel parameters p_c: %f, p_m: %f, tau: %f, bound: %f/%f\n", best_tsp_wheel.p_c, best_tsp_wheel.p_m, best_tsp_wheel.tau, min_tsp_bound + 1, min_tsp_bound);
        printf("Best ATSP wheel parameters p_c: %f, p_m: %f, tau: %f, bound: %f/%f\n", best_atsp_wheel.p_c, best_atsp_wheel.p_m, best_atsp_wheel.tau, min_atsp_bound + 1, min_atsp_bound);

        printf("\n");
    }

    printf("Grid points per dimension: %ld\n", alpha_points);
    printf("Best TSP wheel parameters p_c: %f, p_m: %f, tau: %f, bound: %f/%f\n", best_tsp_wheel.p_c, best_tsp_wheel.p_m, best_tsp_wheel.tau, min_tsp_bound + 1, min_tsp_bound);
    printf("Best ATSP wheel parameters p_c: %f, p_m: %f, tau: %f, bound: %f/%f\n", best_atsp_wheel.p_c, best_atsp_wheel.p_m, best_atsp_wheel.tau, min_atsp_bound + 1, min_atsp_bound);

    free(values_ptr);
    free(points_ptr);
    cudaFree(device_values_ptr);
    cudaFree(device_points_ptr);
}
