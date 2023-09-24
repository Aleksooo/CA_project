#include <iostream>
#include <iomanip>
#include<stdio.h>
#include <array>

#ifndef HANDLER
#define HANDLER

typedef std::array<double, 3> vec;

vec operator+(const vec& lhs, const vec& rhs);
vec operator-(const vec& lhs, const vec& rhs);
vec operator*(const vec& lhs, double val);
vec operator/(const vec& lhs, double val);
std::ostream& operator<<(std::ostream& os, const vec& obj);

#endif
