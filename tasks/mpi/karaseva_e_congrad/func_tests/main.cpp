#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "mpi/karaseva_e_congrad/include/ops_mpi.hpp"

namespace {

std::vector<double> GenerateRandomSPDMatrix(size_t matrix_size, unsigned int seed = 42) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0.1, 1.0);
  std::vector<double> r_matrix(matrix_size * matrix_size);
  for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
    r_matrix[i] = dist(gen);
  }

  std::vector<double> a_matrix(matrix_size * matrix_size, 0.0);
  const auto matrix_size_d = static_cast<double>(matrix_size);

  // Generate symmetric positive-definite matrix
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      for (size_t k = 0; k < matrix_size; ++k) {
        a_matrix[(i * matrix_size) + j] += r_matrix[(k * matrix_size) + i] * r_matrix[(k * matrix_size) + j];
      }
    }
    a_matrix[(i * matrix_size) + i] += matrix_size_d;  // Ensure diagonal dominance
  }
  return a_matrix;
}

std::vector<double> MultiplyMatrixVector(const std::vector<double>& a_matrix, const std::vector<double>& x,
                                         size_t matrix_size) {
  std::vector<double> result(matrix_size, 0.0);
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      result[i] += a_matrix[(i * matrix_size) + j] * x[j];
    }
  }
  return result;
}

}  // namespace

TEST(karaseva_e_congrad_mpi, test_small_matrix_2x2) {
  constexpr size_t kSize = 2;
  constexpr double kTolerance = 1e-10;

  std::vector<double> a_matrix = {4.0, 1.0, 1.0, 3.0};
  std::vector<double> x_expected = {1.0, -2.0};
  auto b_vector = MultiplyMatrixVector(a_matrix, x_expected, kSize);
  std::vector<double> solution(kSize, 0.0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(a_matrix.data()));
  task_data->inputs_count.emplace_back(a_matrix.size());
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(b_vector.data()));
  task_data->inputs_count.emplace_back(b_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(solution.data()));
  task_data->outputs_count.emplace_back(solution.size());

  karaseva_e_congrad_mpi::TestTaskMPI test_task_mpi(task_data);
  ASSERT_TRUE(test_task_mpi.Validation());
  test_task_mpi.PreProcessing();
  test_task_mpi.Run();
  test_task_mpi.PostProcessing();

  for (size_t i = 0; i < kSize; ++i) {
    EXPECT_NEAR(solution[i], x_expected[i], kTolerance);
  }
}