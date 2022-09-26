#include <cstdio>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <chrono>

#define N 1E+8

double integrate(double a, double b, double (*f) (double)) {
    double dx = (b - a) / N;
    double sum = 0;

    for (unsigned i = 0; i < N; i++) {
        sum += f(a+i*dx);
    }

    return dx*sum;
}

int main()
{
    auto f = [](double x) { return x*x; };
    
    auto tm1 = std::chrono::steady_clock::now();
    
    std::cout << integrate(-1, 1, f) << '\n';
    
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - tm1).count() << '\n';
    
    return 0;
}