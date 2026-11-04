// SPDX-License-Identifier: AGPL-3.0-or-later
#include "looplab.h"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    auto blend = [](double x, double y, double b) { return (1-b)*x + b*y; };
    auto linear = [](double x) { return 1.2*x + 0.3; };
    for (int k : {1, 2, 3, 4, 6}) {
        for (double beta : {0.0, 0.5, 1.0}) {
            double x = 0.7;
            const double a = 1 + 0.2/k;
            const double endpoint = std::pow(a, k)*x + 0.3/k*(std::pow(a, k)-1)/(a-1);
            const double expected = beta*linear(x) + (1-beta)*endpoint;
            const double actual = looplab::integrate(x, k, beta, linear, blend);
            assert(std::abs(expected-actual) < 1e-12);
        }
    }
    auto nonlinear = [](double x) { return x + 0.2*std::tanh(x); };
    const double result = looplab::integrate(0.7, 4, 0.5, nonlinear, blend);
    double expected = 0.7;
    for (int i=0; i<4; ++i) { expected += 0.05*std::tanh(expected); }
    expected = 0.5*nonlinear(0.7) + 0.5*expected;
    assert(std::abs(result-expected) < 1e-12);
    assert(std::abs(looplab::integrate(0.7, 4, 1.0, nonlinear, blend)-nonlinear(0.7)) < 1e-12);
    std::cout << "LoopLab integrator: affine closed-form, nonlinear update, and anchor identity passed\n";
}
