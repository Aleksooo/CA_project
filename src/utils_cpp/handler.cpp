#include "handler.hpp"

vec operator+(const vec& lhs, const vec& rhs) {
    return vec({lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]});
}

vec operator-(const vec& lhs, const vec& rhs) {
    return vec({lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]});
}

vec operator*(const vec& lhs, double val) {
    return vec({lhs[0] * val, lhs[1] * val, lhs[2] * val});
}

vec operator/(const vec& lhs, double val) {
    return vec({lhs[0] / val, lhs[1] / val, lhs[2] / val});
}

std::ostream& operator<<(std::ostream& os, const vec& obj) {
    for(auto i : obj) {
        os << std::fixed << std::setw(8) << std::setprecision(3) << i;
    }

    return os;
}

