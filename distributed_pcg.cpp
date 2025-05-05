#include "common.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <utility>
#include <numeric>

#include <Eigen/Sparse>

class CSRMatrix {
  public:
      int nbrow, nbcol;
      std::vector<double> values;
      std::vector<int> col_indices;
      std::vector<int> row_indices;
  
      // Constructor: Build a 1D Poisson matrix with Dirichlet BCs
      CSRMatrix(const int& nrows=0, const int& ncols=0) : nbrow(nrows), nbcol(ncols) {
          row_indices.resize(nbrow + 1);
          for (int i = 0; i < nbrow; ++i) {
              row_indices[i] = values.size();
  
              if (i > 0) {
                  values.push_back(-1.0);
                  col_indices.push_back(i - 1);
              }
  
              values.push_back(2.0);
              col_indices.push_back(i);
  
              if (i + 1 < nbcol) {
                  values.push_back(-1.0);
                  col_indices.push_back(i + 1);
              }
          }
          row_indices[nbrow] = values.size();
      }
  
      // Matrix-vector multiplication y = A * x
      std::vector<double> operator*(const std::vector<double>& x) const {
          assert(x.size() == nbcol);
          std::vector<double> y(nbrow, 0.0);
          for (int i = 0; i < nbrow; ++i) {
              for (int j = row_indices[i]; j < row_indices[i + 1]; ++j) {
                  y[i] += values[j] * x[col_indices[j]];
              }
          }
          return y;
      }
  
      int NbRow() const { return nbrow; }
      int NbCol() const { return nbcol; }
};
  
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

static CSRMatrix A;


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
  A = CSRMatrix(N, N); 
}

/* The preconditioned conjugate gradient method solving Ax = b with tolerance tol.
 * This is the function being evalauted for performance.
 * Note that the starter code only works for 1 rank and it is not efficient.
 */
void CG_Solver::solve(const std::vector<double>& b, std::vector<double>& x, double tol) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the process

  int N = A.NbRow();

  // get the local diagonal block of A
  std::vector<Eigen::Triplet<double>> coefficients;
  for (int i = 0; i < N; ++i) {
    for (int k = A.row_indices[i]; k < A.row_indices[i + 1]; ++k) {
        if (A.col_indices[k] == i)
          coefficients.emplace_back(i, i, A.values[k]);
    }
  }

  // ==========================================
  // UNCOMMENT TO PRINT CHECK COEFFICIENTS
  // ==========================================

   std::cout << "\nTriplets from CSR matrix:\n";
   for (const auto& t : coefficients) {
       std::cout << "(" << t.row() << ", " << t.col() << ") = " << t.value() << "\n";
   }
  

  // compute the Cholesky factorization of the diagonal block for the preconditioner
  Eigen::SparseMatrix<double> B(N, N);
  B.setFromTriplets(coefficients.begin(), coefficients.end());
  Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

  x.assign(N, 0.0);
  std::vector<double> r = b;
  std::vector<double> z = prec(P, r);
  std::vector<double> p = z;
  double rr_loc = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
  double rr;
  MPI_Allreduce(&rr_loc, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double tol_abs = tol * std::sqrt(rr);

  std::vector<double> Ap(N);
  int iter = 0;
  while (std::sqrt(rr) > tol_abs) {
    Ap = A * p;

    double pAp_loc = std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);
    double pAp;
    MPI_Allreduce(&pAp_loc, &pAp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double alpha = rr / pAp;
    for (int i = 0; i < N; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    z = prec(P, r);
    double rr_new_loc = std::inner_product(r.begin(), r.end(), z.begin(), 0.0);
    MPI_Allreduce(&rr_new_loc, &rr, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double beta = rr / rr_loc;

    for (int i = 0; i < N; ++i)
      p[i] = z[i] + beta * p[i]; 

    rr_loc = rr;
    if (rank == 0)
      std::cout << "iter=" << iter << " res=" << std::sqrt(rr) << std::endl;
      ++iter;
  }
 }