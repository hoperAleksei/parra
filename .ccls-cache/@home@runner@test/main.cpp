#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <memory.h>
#include <omp.h>

#define N 100000000

struct result_t {
  double value, milliseconds;
};

result_t run_experiment(double (*integrate)(double, double,
                                            double (*f)(double)),
                        double a, double b, double (*f)(double)) {
  result_t res;
  auto tm1 = std::chrono::steady_clock::now();

  res.value = integrate(-1, 1, f);

  res.milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - tm1)
                         .count();

  return res;
}

double integrate_seq(double a, double b, double (*f)(double)) {
  double dx = (b - a) / N;
  double res = 0;

  for (int i = 0; i < N; i++) {
    res += f(a + i * dx);
  }

  return dx * res;
}

double integrate_par(double a, double b, double (*f)(double)) {
  double dx = (b - a) / N;
  double res = 0;
#pragma omp parallel for reduction(+ : res)
  for (int i = 0; i < N; i++) {
    res += f(a + i * dx);
  }

  return dx * res;
}

double integrate_rr(double a, double b, double (*f)(double)) {
  double dx = (b - a) / N;
  double res = 0;

  unsigned P = omp_get_num_procs();
  double *partical_res;
#pragma omp parallel
  {
    unsigned T = omp_get_num_threads();
    unsigned t = omp_get_thread_num();

#pragma omp single
    { partical_res = (double *)calloc(P, sizeof(double)); }

    for (unsigned R = 0; t + R * T < N; ++R) {
      partical_res[t] += f(a + (t + R * T) * dx);
    }
  }
  for (unsigned t = 0; t < P; ++t)
    res += partical_res[t];

  free(partical_res);

  return dx * res;
}

int main() {
  auto f = [](double x) { return x * x; };
  auto r_seq = run_experiment(integrate_seq, -1, 1, f);
  auto r_par = run_experiment(integrate_par, -1, 1, f);
  auto r_rr = run_experiment(integrate_rr, -1, 1, f);

  std::cout << "Res seq: t=" << r_seq.milliseconds << ", r=" << r_seq.value
            << '\n';
  std::cout << "Res par: t=" << r_par.milliseconds << ", r=" << r_par.value
            << '\n';
  std::cout << "Res rr: t=" << r_rr.milliseconds << ", r=" << r_rr.value
            << '\n';

  return 0;
}