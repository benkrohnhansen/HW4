#include "common.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <utility>

#include <Eigen/Sparse>

class CSRMatrix {
public:
    int nbrow, nbcol;
    std::vector<double> values;
    std::vector<int>    col_indices;
    std::vector<int>    row_indices;

    // 1) Full-matrix constructor (optional; you can keep or remove this)
    CSRMatrix(const int& nrows, const int& ncols, const int& row_start)
      : nbrow(nrows), nbcol(ncols)
    {
      row_indices.resize(nbrow + 1);
      for (int i_local = 0; i_local < nbrow; ++i_local) {
        int ig = row_start + i_local;           // global row index
        row_indices[i_local] = values.size();
    
        // global sub‐diagonal?
        if (ig > 0) {
          values.push_back(-1.0);
          col_indices.push_back(ig - 1);
        }
    
        // diagonal
        values.push_back(2.0);
        col_indices.push_back(ig);
    
        // global super‐diagonal?
        if (ig + 1 < nbcol) {
          values.push_back(-1.0);
          col_indices.push_back(ig + 1);
        }
      }
      row_indices[nbrow] = values.size();
    }
        // Matrix–vector multiply (local rows only)
    std::vector<double> operator*(const std::vector<double>& x) const {
        assert(x.size() == (size_t)nbcol);
        std::vector<double> y(nbrow, 0.0);
        for(int i=0; i<nbrow; ++i) {
            for(int k = row_indices[i]; k < row_indices[i+1]; ++k) {
                y[i] += values[k] * x[col_indices[k]];
            }
        }
        return y;
    }

    int NbRow() const { return nbrow; }
    int NbCol() const { return nbcol; }
};


/// Compute the inner product of two local-length vectors across all ranks
static double global_dot(const std::vector<double>& u,
                         const std::vector<double>& v) {
    assert(u.size() == v.size());
    // 1) local dot-product
    double local = 0.0;
    for (size_t i = 0; i < u.size(); ++i) {
        local += u[i] * v[i];
    }
    // 2) all-reduce to get global dot-product
    double global = 0.0;
    MPI_Allreduce(
      &local,      // sendbuf
      &global,     // recvbuf
      1,           // count
      MPI_DOUBLE,  // datatype
      MPI_SUM,     // op
      MPI_COMM_WORLD
    );
    return global;
}
  
// scalar product (u, v)
double operator,(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  double sp = 0.;
  for(int j = 0; j < u.size(); j++)
    sp += u[j] * v[j];
  return sp; 
}

// addition of two vectors u+v
std::vector<double> operator+(const std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  std::vector<double> w = u;
  for(int j = 0; j < u.size(); j++)
    w[j] += v[j];
  return w;
}

// multiplication of a vector by a scalar a*u
std::vector<double> operator*(const double& a, const std::vector<double>& u){ 
  std::vector<double> w(u.size());
  for(int j = 0; j < w.size(); j++) 
    w[j] = a * u[j];
  return w;
}

// addition assignment operator, add v to u
void operator+=(std::vector<double>& u, const std::vector<double>& v){ 
  assert(u.size() == v.size());
  for(int j = 0; j < u.size(); j++)
    u[j] += v[j];
}

/* block Jacobi preconditioner: perform forward and backward substitution
   using the Cholesky factorization of the local diagonal block computed by Eigen */
std::vector<double> prec(const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P, const std::vector<double>& u){
  Eigen::VectorXd b(u.size());
  for (int i = 0; i < u.size(); i++) 
    b[i] = u[i];
  Eigen::VectorXd xe = P.solve(b);
  std::vector<double> x(u.size());
  for (int i = 0; i < u.size(); i++) 
    x[i] = xe[i];
  return x;
}

CSRMatrix A;


/* N is the size of the matrix, and n is the number of rows assigned per rank.
 * It is your responsibility to generate the input matrix, assuming the ranks are 
 * partitioned rowwise.
 * The input matrix is L + I, where L is the Laplacian of a 1D Possion's equation,
 * and I is the identity matrix.
 * See the constructor of the Matrix structure as an example.
 * The constructor of CG_Solver will not be included in the timing result.
 * Note that the starter code only works for 1 rank and it is not efficient.
 */
CG_Solver::CG_Solver(const int& n, const int& N) {
  // 1) Discover the MPI rank and the total number of ranks
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 2) Compute how many rows we  own
  int n_local = n;

  // 3) Which global rows are they
  int row_start = rank * n_local;
  int row_end = row_start + n_local -1;

  std::cout << "Rank" << rank << " owns rows " << row_start << "-" << row_end << "\n";

  // 4) Pass those into the CSR  builder so it  only builds that  slice
  A = CSRMatrix(n_local, N, row_start);
}

/* The preconditioned conjugate gradient method solving Ax = b with tolerance tol.
 * This is the function being evalauted for performance.
 * Note that the starter code only works for 1 rank and it is not efficient.
 */
void CG_Solver::solve(const std::vector<double>& b,
                      std::vector<double>& x,
                      double tol) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n_local = A.NbRow();                 // rows per rank
    int row_start = rank * n_local;

  // p_global: concatenation of all ranks' p (length = n_local * size)
  // Ap_local: to cache our local y = A * p_global (length = n_local)
  std::vector<double> p_global(n_local * size);
  std::vector<double> Ap_local(n_local);

  // Build the local diagonal block for the block‐Jacobi preconditioner
  std::vector<Eigen::Triplet<double>> coefficients;
  coefficients.reserve(3 * n_local);
  for (int i_local = 0; i_local < n_local; ++i_local) {
    // scan your CSR row i_local
    for (int idx = A.row_indices[i_local]; idx < A.row_indices[i_local+1]; ++idx) {
      int global_col = A.col_indices[idx];       // e.g. ig-1, ig, or ig+1
      int local_col  = global_col - row_start;   // now in [0..n_local-1]
      double val     = A.values[idx];
      coefficients.emplace_back(i_local, local_col, val);
    }
  }
  Eigen::SparseMatrix<double> B(n_local, n_local);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  // Initial norms and vectors
  double norm_b    = std::sqrt(global_dot(b, b)); 
const double eps = tol * norm_b;
std::vector<double> r = b, z = prec(P, b), p = z;
double rz = global_dot(r, z);
int num_it = 0;

// Main PCG loop
while (true) {
  // 1) Gather global search direction
  MPI_Allgather(
    p.data(),        n_local, MPI_DOUBLE,
    p_global.data(), n_local, MPI_DOUBLE,
    MPI_COMM_WORLD
  );

  // 2) Local mat–vec: Ap_local = A * p_global
  Ap_local = A * p_global;

  // 3) Compute alpha = (r,z) / (p,Ap)
  double pAp   = global_dot(p, Ap_local);
  double alpha = rz / pAp;

  // 4) Update x and r locally
  for (int i = 0; i < n_local; ++i) {
    x[i] += alpha * p[i];
    r[i] -= alpha * Ap_local[i];
  }

  // 5) Apply preconditioner locally
  z = prec(P, r);

  // 6) Compute new (r,z), print, and test convergence
  double rz_new = global_dot(r, z);
  num_it++;
  if (rank == 0) {
    double res_norm = std::sqrt(global_dot(r, r));
    std::cout << "iteration: " << num_it
              << "\tresidual:  " << res_norm << "\n";
  }
  if (std::sqrt(rz_new) < eps)
    break;

  // 7) Update search direction and rz for next iteration
  double beta = rz_new / rz;
  for (int i = 0; i < n_local; ++i)
    p[i] = z[i] + beta * p[i];
  rz = rz_new;
}
}
