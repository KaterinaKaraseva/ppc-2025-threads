#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/malyshev_v_radix_sort/include/ops_stl.hpp"

std::vector<double> malyshev_v_radix_sort_stl::GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(malyshev_v_radix_sort_all, test_pipeline_run) {
  int global_vector_size = 3000000;
  std::vector<double> global_vector = malyshev_v_radix_sort_stl::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  auto test_task_stl = std::make_shared<malyshev_v_radix_sort_stl::TestTaskSTL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (std::vector<double>::size_type i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}

TEST(malyshev_v_radix_sort_all, test_task_run) {
  int global_vector_size = 3000000;
  std::vector<double> global_vector = malyshev_v_radix_sort_stl::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  task_data->outputs_count.emplace_back(result.size());
  auto test_task_stl = std::make_shared<malyshev_v_radix_sort_stl::TestTaskSTL>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  for (std::vector<double>::size_type i = 1; i < result.size(); i++) {
    ASSERT_LE(result[i - 1], result[i]);
  }
}