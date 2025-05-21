#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/karaseva_e_congrad/include/ops_mpi.hpp"

TEST(karaseva_e_congrad_mpi, test_pipeline_run) {
  constexpr int kCount = 500;

  // Create data
  std::vector<double> A(kCount * kCount, 0.0);
  std::vector<double> b(kCount, 1.0);
  std::vector<double> x(kCount, 0.0);

  // Initialize A as identity matrix
  for (size_t i = 0; i < kCount; i++) {
    A[i * kCount + i] = 1.0;
  }

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_mpi->inputs_count.emplace_back(A.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_mpi->inputs_count.emplace_back(b.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_mpi->outputs_count.emplace_back(x.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_congrad_mpi::TestTaskMPI>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  // Verify the result on root process
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (size_t i = 0; i < kCount; ++i) {
      EXPECT_NEAR(x[i], b[i], 1e-6);
    }
  }
}

TEST(karaseva_e_congrad_mpi, test_task_run) {
  constexpr int kCount = 500;

  // Create data
  std::vector<double> A(kCount * kCount, 0.0);
  std::vector<double> b(kCount, 1.0);
  std::vector<double> x(kCount, 0.0);

  // Initialize A as identity matrix
  for (size_t i = 0; i < kCount; i++) {
    A[i * kCount + i] = 1.0;
  }

  // Create task_data
  auto task_data_mpi = std::make_shared<ppc::core::TaskData>();
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  task_data_mpi->inputs_count.emplace_back(A.size());
  task_data_mpi->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  task_data_mpi->inputs_count.emplace_back(b.size());
  task_data_mpi->outputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
  task_data_mpi->outputs_count.emplace_back(x.size());

  // Create Task
  auto test_task_mpi = std::make_shared<karaseva_e_congrad_mpi::TestTaskMPI>(task_data_mpi);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);
  perf_analyzer->TaskRun(perf_attr, perf_results);

  // Verify the result on root process
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (size_t i = 0; i < kCount; ++i) {
      EXPECT_NEAR(x[i], b[i], 1e-6);
    }
  }
}