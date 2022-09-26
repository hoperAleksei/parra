#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

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

int main() {
  auto f = [](double x) { return x * x; };
  auto r_seq = run_experiment(integrate_seq, -1, 1, f);
  auto r_par = run_experiment(integrate_par, -1, 1, f);

  std::cout << "Res seq: t=" << r_seq.milliseconds << ", r=" << r_seq.value
            << '\n';
  std::cout << "Res par: t=" << r_par.milliseconds << ", r=" << r_par.value
            << '\n';

  return 0;
}