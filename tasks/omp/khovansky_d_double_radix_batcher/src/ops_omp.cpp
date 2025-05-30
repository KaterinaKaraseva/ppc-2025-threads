#include "omp/khovansky_d_double_radix_batcher/include/ops_omp.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace khovansky_d_double_radix_batcher_omp {
namespace {
uint64_t EncodeDoubleToUint64(double value) {
  uint64_t bit_representation = 0;
  std::memcpy(&bit_representation, &value, sizeof(value));

  if ((bit_representation >> 63) != 0) {
    return ~bit_representation;
  }
  return bit_representation ^ (1ULL << 63);
}

double DecodeUint64ToDouble(uint64_t encoded) {
  if ((encoded >> 63) != 0) {
    encoded ^= (1ULL << 63);
  } else {
    encoded = ~encoded;
  }

  double result = 0.0;
  std::memcpy(&result, &encoded, sizeof(result));
  return result;
}

void RadixSort(std::vector<uint64_t>& array) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size(), 0);
  std::vector<int> frequency(bucket_count, 0);

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    std::vector<int> local_frequency(bucket_count, 0);

#pragma omp parallel
    {
      std::vector<int> private_frequency(bucket_count, 0);

#pragma omp for nowait
      for (int64_t i = 0; i < static_cast<int64_t>(array.size()); i++) {
        auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
        private_frequency[bucket]++;
      }

#pragma omp critical
      for (int i = 0; i < bucket_count; i++) {
        local_frequency[i] += private_frequency[i];
      }
    }

    for (int i = 1; i < bucket_count; i++) {
      local_frequency[i] += local_frequency[i - 1];
    }

    for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
      auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
      buffer[--local_frequency[bucket]] = array[i];
    }

    array.swap(buffer);
  }
}

void OddEvenMergeSort(std::vector<uint64_t>& array, int left, int right) {
  if (right - left <= 1) {
    return;
  }

  int middle = left + ((right - left) / 2);

#pragma omp parallel sections
  {
#pragma omp section
    OddEvenMergeSort(array, left, middle);
#pragma omp section
    OddEvenMergeSort(array, middle, right);
  }

#pragma omp parallel for
  for (int i = left; i < right - 1; i += 2) {
    if (array[i] > array[i + 1]) {
      std::swap(array[i], array[i + 1]);
    }
  }
}

void RadixBatcherSort(std::vector<double>& data) {
  std::vector<uint64_t> transformed_data(data.size(), 0);

#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(data.size()); i++) {
    transformed_data[i] = EncodeDoubleToUint64(data[i]);
  }

  RadixSort(transformed_data);
  OddEvenMergeSort(transformed_data, 0, static_cast<int>(transformed_data.size()));

#pragma omp parallel for
  for (int64_t i = 0; i < static_cast<int64_t>(data.size()); i++) {
    data[i] = DecodeUint64ToDouble(transformed_data[i]);
  }
}
}  // namespace
}  // namespace khovansky_d_double_radix_batcher_omp

bool khovansky_d_double_radix_batcher_omp::RadixOMP::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  unsigned int input_size = task_data->inputs_count[0];
  unsigned int output_size = task_data->outputs_count[0];

  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool khovansky_d_double_radix_batcher_omp::RadixOMP::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] < 2) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khovansky_d_double_radix_batcher_omp::RadixOMP::RunImpl() {
  output_ = input_;
  khovansky_d_double_radix_batcher_omp::RadixBatcherSort(output_);
  return true;
}

bool khovansky_d_double_radix_batcher_omp::RadixOMP::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }

  return true;
}