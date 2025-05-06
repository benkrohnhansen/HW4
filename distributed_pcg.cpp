#include "common.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <Eigen/Sparse>

class CSRMatrix {
 public:
  int nbrow, nbcol, start_row;
  std::vector<double> values;
  std::vector<int> col_indices;
  std::vector<int> row_indices;

  CSRMatrix(int nrows = 0, int ncols = 0, int start = 0)
      : nbrow(nrows), nbcol(ncols), start_row(start) {
    row_indices.resize(nbrow + 1);
    for (int i = 0; i < nbrow; ++i) {
      int global_row = start_row + i;
      row_indices[i] = values.size();

      if (global_row > 0) {
        values.push_back(-1.0);
        col_indices.push_back(global_row - 1);
      }

      values.push_back(2.0);
      col_indices.push_back(global_row);

      if (global_row + 1 < nbcol) {
        values.push_back(-1.0);
        col_indices.push_back(global_row + 1);
      }
    }
    row_indices[nbrow] = values.size();

    // Debug: Print matrix values for rank 6
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 6) {
        std::cout << "Rank 6 CSR Matrix Values: " << std::endl;
        for (size_t i = 0; i < values.size(); ++i) {
            std::cout << "Row " << i << ": Value = " << values[i] 
                      << ", Column Index = " << col_indices[i] << std::endl;
        }
    }
  }

  std::vector<double> operator*(const std::vector<double>& x_local) const {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = nbrow;
    std::vector<double> x_extended(n + 2);  // ghost_left + local + ghost_right

    for (int i = 0; i < n; ++i) x_extended[i + 1] = x_local[i];

    MPI_Request requests[4];
    int req_count = 0;
    double left_send = x_local[0], right_send = x_local[n - 1];
    double left_recv = 0.0, right_recv = 0.0;

    if (rank > 0) {
      MPI_Isend(&left_send, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
      MPI_Irecv(&left_recv, 1, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }
    if (rank < size - 1) {
      MPI_Isend(&right_send, 1, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, &requests[req_count++]);
      MPI_Irecv(&right_recv, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

    // Debug: Print communication details for rank 6
    if (rank == 6) {
        std::cout << "Rank 6 sent left: " << left_send << ", received left: " << left_recv << std::endl;
        std::cout << "Rank 6 sent right: " << right_send << ", received right: " << right_recv << std::endl;
    }

    if (rank > 0) x_extended[0] = left_recv;
    if (rank < size - 1) x_extended[n + 1] = right_recv;

    std::vector<double> y(nbrow, 0.0);
    for (int i = 0; i < nbrow; ++i) {
      for (int j = row_indices[i]; j < row_indices[i + 1]; ++j) {
        int global_col = col_indices[j];
        int idx = global_col - start_row + 1;
        y[i] += values[j] * x_extended[idx];
      }
    }
    return y;
  }
};

// Vector utilities and preconditioned CG solver omitted for brevity

void CG_Solver::solve(const std::vector<double>& b, std::vector<double>& x, double tol) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = A.NbRow();
  
  Eigen::SparseMatrix<double> B(n, n);
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  const double epsilon = tol * std::sqrt(dot(b, b));
  x.assign(n, 0.);
  std::vector<double> r = b, z = prec(P, b), p = z;
  double alpha = 0., beta = 0.;
  double res = std::sqrt(dot(r, r));

  int num_it = 0;

  while (res >= epsilon) {
    std::vector<double> Ap = A * p;
    alpha = dot(r, z) / dot(p, Ap);
    x += alpha * p;
    r += -alpha * Ap;
    z = prec(P, r);
    beta = dot(r, z) / (alpha * dot(p, Ap));
    p = z + beta * p;
    res = std::sqrt(dot(r, r));

    // Debug: Print residual values for rank 6
    if (rank == 6) {
        std::cout << "Rank 6, Iteration " << num_it << " | r * r: " << dot(r, r) << " | res: " << res << std::endl;
    }

    num_it++;
    if (rank == 0 && !(num_it % 1)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << res << "\n";
    }
  }
}
