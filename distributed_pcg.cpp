#include "common.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <iomanip>

#include <Eigen/Sparse>

class CSRMatrix {
public:
    int nbrow, nbcol;
    std::vector<double> values;
    std::vector<int>    col_indices;
    std::vector<int>    row_indices;

    CSRMatrix(const int& nrows = 0, const int& ncols = 0)
      : nbrow(nrows), nbcol(ncols)
    {
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

double operator,(
    const std::vector<double>& u,
    const std::vector<double>& v)
{
    assert(u.size() == v.size());
    double sp = 0.0;
    for (size_t j = 0; j < u.size(); ++j)
        sp += u[j] * v[j];
    return sp;
}

std::vector<double> operator+(
    const std::vector<double>& u,
    const std::vector<double>& v)
{
    assert(u.size() == v.size());
    std::vector<double> w = u;
    for (size_t j = 0; j < u.size(); ++j)
        w[j] += v[j];
    return w;
}

std::vector<double> operator*(
    const double& a,
    const std::vector<double>& u)
{
    std::vector<double> w(u.size());
    for (size_t j = 0; j < u.size(); ++j)
        w[j] = a * u[j];
    return w;
}

void operator+=(
    std::vector<double>& u,
    const std::vector<double>& v)
{
    assert(u.size() == v.size());
    for (size_t j = 0; j < u.size(); ++j)
        u[j] += v[j];
}

std::vector<double> prec(
    const Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>>& P,
    const std::vector<double>& u)
{
    Eigen::VectorXd b(u.size());
    for (size_t i = 0; i < u.size(); ++i)
        b[i] = u[i];
    Eigen::VectorXd xe = P.solve(b);
    std::vector<double> x(u.size());
    for (size_t i = 0; i < u.size(); ++i)
        x[i] = xe[i];
    return x;
}

static CSRMatrix A;

CG_Solver::CG_Solver(const int& n, const int& N) {
    A = CSRMatrix(N, N);
}

void CG_Solver::solve(
    const std::vector<double>& b,
    std::vector<double>& x,
    double tol)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = A.NbRow();

    std::vector<Eigen::Triplet<double>> coeffs;
    for (int i = 0; i < N; ++i) {
        for (int k = A.row_indices[i]; k < A.row_indices[i + 1]; ++k) {
            if (A.col_indices[k] == i)
                coeffs.emplace_back(i, i, A.values[k]);
        }
    }
    Eigen::SparseMatrix<double> B(N, N);
    B.setFromTriplets(coeffs.begin(), coeffs.end());
    Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> P(B);

    x.assign(N, 0.0);
    std::vector<double> r = b;
    std::vector<double> z = prec(P, r);
    std::vector<double> p = z;
    std::vector<double> Ap(N);

    double epsilon = tol * std::sqrt((r, r));
    int num_it = 0;
    double t_start = MPI_Wtime();

    while (true) {
        Ap = A * p;

        double alpha = (r, z) / (p, Ap);
        x += alpha * p;
        r += -alpha * Ap;

        z = prec(P, r);
        double beta = (r, z) / (alpha * (p, Ap));
        p = z + beta * p;

        ++num_it;
        double rr = std::sqrt((r, r));
        if (rank == 0 && num_it <= 2) {
            std::cout
              << "iteration: " << num_it
              << "    residual:  "
              << std::scientific << std::setprecision(7)
              << rr << "\n";
        }
        if (rr < epsilon) break;
    }
    double t_end = MPI_Wtime();
    if (rank == 0) {
        std::cout
          << "Time for CG of size " << N
          << " with " << size << " rank(s): "
          << std::fixed << std::setprecision(6)
          << (t_end - t_start) << " seconds.\n";

        auto Ax = A * x;
        std::vector<double> diff(N);
        for (int i = 0; i < N; ++i)
            diff[i] = Ax[i] - b[i];

        double num = std::sqrt(
          std::inner_product(
            diff.begin(), diff.end(), diff.begin(), 0.0));
        double den = std::sqrt(
          std::inner_product(
            b.begin(), b.end(), b.begin(), 0.0));
        std::cout
          << "|Ax - b| / |b| = "
          << std::scientific << std::setprecision(5)
          << (num/den) << "\n";
    }
}
