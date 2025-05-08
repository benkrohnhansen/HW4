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

  int NbRow() const { return nbrow; }
  int NbCol() const { return nbcol; }
};

// Vector utilities

double dot(const std::vector<double>& u, const std::vector<double>& v) {
  assert(u.size() == v.size());
  double local_sum = 0.;
  for (size_t j = 0; j < u.size(); ++j)
    local_sum += u[j] * v[j];
  double global_sum;
  MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return global_sum;
}

std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v) {
  assert(u.size() == v.size());
  std::vector<double> w = u;
  for (size_t j = 0; j < u.size(); ++j)
    w[j] += v[j];
  return w;
}

std::vector<double> operator*(double a, const std::vector<double>& u) {
  std::vector<double> w(u.size());
  for (size_t j = 0; j < w.size(); ++j)
    w[j] = a * u[j];
  return w;
}

void operator+=(std::vector<double>& u, const std::vector<double>& v) {
  assert(u.size() == v.size());
  for (size_t j = 0; j < u.size(); ++j)
    u[j] += v[j];
}

std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u) {
  Eigen::VectorXd b(u.size());
  for (size_t i = 0; i < u.size(); ++i) b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b);
  std::vector<double> x(u.size());
  for (size_t i = 0; i < u.size(); ++i) x[i] = xe[i];
  return x;
}

static CSRMatrix A;

CG_Solver::CG_Solver(const int& n, const int& N) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int start_row = rank * n;
  A = CSRMatrix(n, N, start_row);
}

void CG_Solver::solve(const std::vector<double>& b, std::vector<double>& x, double tol) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int n = A.NbRow();
  std::vector<Eigen::Triplet<double>> coefficients;
  for (int row = 0; row < A.NbRow(); ++row) {
    for (int idx = A.row_indices[row]; idx < A.row_indices[row + 1]; ++idx) {
      int col = A.col_indices[idx];
      if (col >= rank * n && col < (rank + 1) * n) {
        coefficients.emplace_back(row, col - rank * n, A.values[idx]);
      }
    }
  }
  

  Eigen::SparseMatrix<double> B(n, n);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  const double epsilon = tol * std::sqrt(dot(b, b));
  x.assign(n, 0.);
  std::vector<double> r = b, z = prec(P, b), p = z;
  double alpha = 0., beta = 0.;
  double res = std::sqrt(dot(r, r));
  
  double p_dot_Ap = 0., r_dot_z = 0.; // intermediates - Ben

  int num_it = 0;

  double t_start = MPI_Wtime();

  while (res >= epsilon) {
    std::vector<double> Ap = A * p;
    p_dot_Ap = dot(p, Ap); 
    r_dot_z = dot(r, z);
    alpha = r_dot_z / p_dot_Ap;
    x += alpha * p;
    r += -alpha * Ap;
    z = prec(P, r);
    beta = dot(r, z) / r_dot_z;
    p = z + beta * p;
    res = std::sqrt(dot(r, r));

    num_it++;
    if (rank == 0 && !(num_it % 1)) {
      std::cout << "iteration: " << num_it << "\t";
      std::cout << "residual:  " << res << "\n";
    }
  }

  // === ⏱️ End timing
  double t_end = MPI_Wtime();
  double t_elapsed = t_end - t_start;

  // Report total and per-iteration time
  if (rank == 0) {
    std::cout << "\n==== PCG Timing Summary ====\n";
    std::cout << "Total iterations: " << num_it << "\n";
    std::cout << "Total time:       " << t_elapsed << " seconds\n";
    std::cout << "Time per iter:    " << t_elapsed / num_it << " seconds/iter\n";
  }
}
