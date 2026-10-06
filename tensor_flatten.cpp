#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Dimensions {
    std::size_t batch;
    std::size_t channels;
    std::size_t height;
    std::size_t width;
};

// Safe multiplication with overflow check
std::size_t checkedProduct(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("Tensor dimensions exceed memory address limits.");
    }
    return a * b;
}

std::size_t tensorSize(const Dimensions& dims) {
    return checkedProduct(
        checkedProduct(checkedProduct(dims.batch, dims.channels), dims.height),
        dims.width);
}

/**
 * @brief Forward Transformation: BCHW -> B x (CHW)
 * Converts a 4D tensor I (size B x C x H x W) to a 2D matrix F (size B x (CHW))
 * using explicit row-major address indexing: k = c * (H * W) + h * W + w.
 */
std::vector<std::vector<double>> flattenBCHW_to_2D(
    const std::vector<double>& input,
    const Dimensions& dims) {
    if (input.size() != tensorSize(dims)) {
        throw std::invalid_argument("Input size does not match BCHW dimensions.");
    }

    const std::size_t chw = checkedProduct(
        checkedProduct(dims.channels, dims.height), dims.width);
    const std::size_t hw = checkedProduct(dims.height, dims.width);

    // Explicit 2D Matrix of size [B][CHW]
    std::vector<std::vector<double>> flattened(dims.batch, std::vector<double>(chw));

    for (std::size_t b = 0; b < dims.batch; ++b) {
        for (std::size_t c = 0; c < dims.channels; ++c) {
            for (std::size_t h = 0; h < dims.height; ++h) {
                for (std::size_t w = 0; w < dims.width; ++w) {
                    const std::size_t inputIndex =
                        b * chw + c * hw + h * dims.width + w;
                    // Compute column index k in the flattened 2D tensor
                    const std::size_t k = c * hw + h * dims.width + w;
                    flattened[b][k] = input[inputIndex];
                }
            }
        }
    }

    return flattened;
}

/**
 * @brief Reverse Transformation: B x (CHW) -> BCHW
 * Reconstructs a 4D tensor I_hat (size B x C x H x W) from a 2D matrix F (size B x (CHW))
 * by decoding 1D column coordinate k into 3D (c, h, w) via quotient and modulo arithmetic.
 */
std::vector<double> reconstruct2D_to_BCHW(
    const std::vector<std::vector<double>>& flattened,
    const Dimensions& dims) {
    const std::size_t chw = checkedProduct(
        checkedProduct(dims.channels, dims.height), dims.width);
    const std::size_t hw = checkedProduct(dims.height, dims.width);

    if (flattened.size() != dims.batch) {
        throw std::invalid_argument("Flattened row count does not match Batch dimension.");
    }

    std::vector<double> reconstructed(tensorSize(dims));

    for (std::size_t b = 0; b < dims.batch; ++b) {
        if (flattened[b].size() != chw) {
            throw std::invalid_argument("Flattened column count does not match CHW dimension.");
        }

        for (std::size_t k = 0; k < chw; ++k) {
            // Inverse coordinate decoding via integer division and modulo
            const std::size_t c = k / hw;
            const std::size_t rem = k % hw;
            const std::size_t h = rem / dims.width;
            const std::size_t w = rem % dims.width;

            const std::size_t outputIndex =
                b * chw + c * hw + h * dims.width + w;
            reconstructed[outputIndex] = flattened[b][k];
        }
    }

    return reconstructed;
}

// Generate deterministic synthetic test tensor
std::vector<double> makeSyntheticTensor(const Dimensions& dims) {
    std::vector<double> input(tensorSize(dims));
    const std::size_t chw = checkedProduct(
        checkedProduct(dims.channels, dims.height), dims.width);
    const std::size_t hw = checkedProduct(dims.height, dims.width);

    for (std::size_t b = 0; b < dims.batch; ++b) {
        for (std::size_t c = 0; c < dims.channels; ++c) {
            for (std::size_t h = 0; h < dims.height; ++h) {
                for (std::size_t w = 0; w < dims.width; ++w) {
                    const std::size_t index =
                        b * chw + c * hw + h * dims.width + w;
                    input[index] = static_cast<double>(
                        ((b + 1) * 1000000 + (c + 1) * 10000 +
                         (h + 1) * 100 + (w + 1)) % 1000003) /
                        1000.0;
                }
            }
        }
    }

    return input;
}

struct ErrorMeasures {
    double maximumAbsoluteError;
    double meanAbsoluteError;
};

ErrorMeasures measureError(
    const std::vector<double>& original,
    const std::vector<double>& reconstructed) {
    if (original.size() != reconstructed.size() || original.empty()) {
        throw std::invalid_argument(
            "Error measurement requires equally sized, non-empty tensors.");
    }

    double maximumAbsoluteError = 0.0;
    double totalAbsoluteError = 0.0;
    for (std::size_t i = 0; i < original.size(); ++i) {
        const double absoluteError = std::abs(original[i] - reconstructed[i]);
        if (absoluteError > maximumAbsoluteError) {
            maximumAbsoluteError = absoluteError;
        }
        totalAbsoluteError += absoluteError;
    }

    return {maximumAbsoluteError, totalAbsoluteError / original.size()};
}

void printManualExample() {
    std::cout << "================================================================================\n";
    std::cout << "MANUAL INDEXING EXAMPLE (B=1, C=2, H=2, W=3)\n";
    std::cout << "================================================================================\n";

    const Dimensions dims{1, 2, 2, 3};
    std::vector<double> input(tensorSize(dims));
    const std::size_t hw = dims.height * dims.width;

    for (std::size_t c = 0; c < dims.channels; ++c) {
        for (std::size_t h = 0; h < dims.height; ++h) {
            for (std::size_t w = 0; w < dims.width; ++w) {
                const std::size_t index = c * hw + h * dims.width + w;
                input[index] = static_cast<double>(c * 100 + h * 10 + w);
            }
        }
    }

    const std::vector<std::vector<double>> flattened = flattenBCHW_to_2D(input, dims);
    const std::vector<double> reconstructed = reconstruct2D_to_BCHW(flattened, dims);
    const ErrorMeasures errors = measureError(input, reconstructed);

    std::cout << "  I[0,0,0,0] -> F[0][0]  = " << flattened[0][0] << '\n';
    std::cout << "  I[0,0,1,2] -> F[0][5]  = " << flattened[0][5] << '\n';
    std::cout << "  I[0,1,0,0] -> F[0][6]  = " << flattened[0][6] << '\n';
    std::cout << "  I[0,1,1,2] -> F[0][11] = " << flattened[0][11] << '\n';
    std::cout << "  Flattened row: [";
    for (std::size_t i = 0; i < flattened[0].size(); ++i) {
        std::cout << (i == 0 ? "" : ", ") << flattened[0][i];
    }
    std::cout << "]\n";
    std::cout << "  Manual verification: Emax = " << errors.maximumAbsoluteError 
              << ", MAE = " << errors.meanAbsoluteError << "\n\n";
}

struct ExperimentCase {
    std::string name;
    Dimensions dims;
};

int main() {
    try {
        printManualExample();

        const ExperimentCase experiments[] = {
            {"Synthetic (MNIST shape)", {1,   1, 28, 28}},
            {"Synthetic (CIFAR shape)", {1,   3, 32, 32}},
            {"Synthetic fmap1",  {2,  16, 16, 16}},
            {"Synthetic fmap2",  {2,  64, 16, 16}},
            {"Synthetic fmap3",  {2, 128, 16, 16}},
            {"Synthetic fmap4",  {2, 500, 16, 16}}
        };

        std::cout << "================================================================================\n";
        std::cout << "EXPERIMENTAL RESULTS TABLE (Table 1)\n";
        std::cout << "================================================================================\n";
        std::cout << std::left << std::setw(22) << "Dataset/Input"
                  << std::right << std::setw(6) << "B"
                  << std::setw(7) << "C"
                  << std::setw(7) << "H"
                  << std::setw(7) << "W"
                  << std::setw(22) << "Emax"
                  << std::setw(22) << "MAE" << '\n';
        std::cout << std::string(93, '-') << '\n';
        std::cout << std::fixed << std::setprecision(12);

        for (const auto& exp : experiments) {
            const std::vector<double> input = makeSyntheticTensor(exp.dims);
            const std::vector<std::vector<double>> flattened =
                flattenBCHW_to_2D(input, exp.dims);
            const std::vector<double> reconstructed =
                reconstruct2D_to_BCHW(flattened, exp.dims);
            const ErrorMeasures errors = measureError(input, reconstructed);

            std::cout << std::left << std::setw(22) << exp.name
                      << std::right << std::setw(6) << exp.dims.batch
                      << std::setw(7) << exp.dims.channels
                      << std::setw(7) << exp.dims.height
                      << std::setw(7) << exp.dims.width
                      << std::setw(22) << errors.maximumAbsoluteError
                      << std::setw(22) << errors.meanAbsoluteError << '\n';
        }
        std::cout << std::string(93, '-') << '\n';

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
