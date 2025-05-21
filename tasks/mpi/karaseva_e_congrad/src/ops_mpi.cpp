#include "mpi/karaseva_e_congrad/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/operations.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"
#include "mpi.h"

namespace karaseva_e_congrad_mpi {

bool TestTaskMPI::PreProcessingImpl() {
  int rank = world_.rank();
  int size = world_.size();

  if (rank == 0) {
    global_size_ = task_data->inputs_count[1];
    auto* a_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* b_ptr = reinterpret_cast<double*>(task_data->inputs[1]);

    const size_t rows_per_proc = global_size_ / static_cast<size_t>(size);
    const size_t remainder = global_size_ % static_cast<size_t>(size);

    for (int proc = 0; proc < size; ++proc) {
      const size_t proc_idx = static_cast<size_t>(proc);
      const size_t start_row = proc_idx * rows_per_proc + std::min(proc_idx, remainder);
      const size_t end_row = (proc_idx + 1) * rows_per_proc + std::min(proc_idx + 1, remainder);
      const size_t num_rows = end_row - start_row;

      // Prepare local matrix part
      std::vector<double> local_A(num_rows * global_size_);
      for (size_t r = 0; r < num_rows; ++r) {
        const size_t global_row = start_row + r;
        std::copy(a_ptr + global_row * global_size_, a_ptr + (global_row + 1) * global_size_,
                  local_A.begin() + r * global_size_);
      }

      // Prepare local vector part
      std::vector<double> local_b(num_rows);
      for (size_t r = 0; r < num_rows; ++r) {
        local_b[r] = b_ptr[start_row + r];
      }

      if (proc == 0) {
        A_ = std::move(local_A);
        b_ = std::move(local_b);
        local_size_ = num_rows;
      } else {
        world_.send(proc, 0, local_A);
        world_.send(proc, 0, local_b);
      }
    }
  } else {
    world_.recv(0, 0, A_);
    world_.recv(0, 0, b_);
    local_size_ = b_.size();
  }

  x_.resize(local_size_, 0.0);
  return true;
}

bool TestTaskMPI::ValidationImpl() {
  bool valid = true;
  if (world_.rank() == 0) {
    valid = (task_data->inputs_count[0] == task_data->inputs_count[1] * task_data->inputs_count[1]) &&
            (task_data->outputs_count[0] == task_data->inputs_count[1]);
  }
  boost::mpi::broadcast(world_, valid, 0);
  return valid;
}

bool TestTaskMPI::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  std::vector<double> r(local_size_);   // Local residual
  std::vector<double> p(local_size_);   // Local search direction
  std::vector<double> ap(local_size_);  // Local part of A*p

  // Initialize residual and search direction
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(local_size_); ++i) {
    r[i] = b_[i];
    p[i] = r[i];
  }

  // Compute initial residual norm
  double local_rs_old = 0.0;
#pragma omp parallel for reduction(+ : local_rs_old)
  for (int i = 0; i < static_cast<int>(local_size_); ++i) {
    local_rs_old += r[i] * r[i];
  }
  double rs_old = 0.0;
  boost::mpi::all_reduce(world_, local_rs_old, rs_old, std::plus<>());

  constexpr double tolerance = 1e-10;
  const size_t max_iterations = global_size_;

  // Main conjugate gradient loop
  for (size_t it = 0; it < max_iterations; ++it) {
    // Alternative to all_gatherv: gather + broadcast
    std::vector<double> global_p;
    if (rank == 0) {
      global_p.resize(global_size_);
    }

    // Gather parts to root
    boost::mpi::gather(world_, p.data(), static_cast<int>(local_size_), global_p, 0);

    // Broadcast full vector to all processes
    boost::mpi::broadcast(world_, global_p, 0);

    // Compute local part of matrix-vector product
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(local_size_); ++i) {
      double sum = 0.0;
      for (size_t j = 0; j < global_size_; ++j) {
        sum += A_[i * global_size_ + j] * global_p[j];
      }
      ap[i] = sum;
    }

    // Compute p^T * A * p
    double local_p_ap = 0.0;
#pragma omp parallel for reduction(+ : local_p_ap)
    for (int i = 0; i < static_cast<int>(local_size_); ++i) {
      local_p_ap += p[i] * ap[i];
    }
    double p_ap = 0.0;
    boost::mpi::all_reduce(world_, local_p_ap, p_ap, std::plus<>());

    if (std::fabs(p_ap) < 1e-15) break;
    const double alpha = rs_old / p_ap;

    // Update solution and residual
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(local_size_); ++i) {
      x_[i] += alpha * p[i];
      r[i] -= alpha * ap[i];
    }

    // Compute new residual norm
    double local_rs_new = 0.0;
#pragma omp parallel for reduction(+ : local_rs_new)
    for (int i = 0; i < static_cast<int>(local_size_); ++i) {
      local_rs_new += r[i] * r[i];
    }
    double rs_new = 0.0;
    boost::mpi::all_reduce(world_, local_rs_new, rs_new, std::plus<>());

    if (rs_new < tolerance * tolerance) break;

    // Update search direction
    const double beta = rs_new / rs_old;
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(local_size_); ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }

  // Gather final solution to root process
  std::vector<double> global_x;
  if (rank == 0) {
    global_x.resize(global_size_);
  }

  boost::mpi::gather(world_, x_.data(), static_cast<int>(local_size_), global_x, 0);

  if (rank == 0) {
    auto* output = reinterpret_cast<double*>(task_data->outputs[0]);
    std::copy(global_x.begin(), global_x.end(), output);
  }

  return true;
}

bool TestTaskMPI::PostProcessingImpl() { return true; }

}  // namespace karaseva_e_congrad_mpi