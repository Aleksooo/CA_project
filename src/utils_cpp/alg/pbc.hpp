#include <cmath>
#include "../handler.hpp"

double dot(vec a, vec b);
vec cross(vec a, vec b);

double norm2(vec point);
double norm(vec point);

double dist2(vec point1, vec point2);
double dist(vec point1, vec point2);

int sign(double x);
vec delta_pbc(vec point1, vec point2, vec box);
double dist2_pbc(vec point1, vec point2, vec box);
double dist_pbc(vec point1, vec point2, vec box);

vec rotate_point(vec point, double psi, double theta, double phi);
