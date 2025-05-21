#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace karaseva_e_congrad_mpi {

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)), world_() {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> A_;           // Local part of coefficient matrix
  std::vector<double> b_;           // Local part of right-hand side vector
  std::vector<double> x_;           // Local part of solution vector
  size_t global_size_{};            // Global system size
  size_t local_size_{};             // Local rows count
  boost::mpi::communicator world_;  // MPI communicator
};

}  // namespace karaseva_e_congrad_mpi