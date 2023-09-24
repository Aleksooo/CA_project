#include "pbc.hpp"

double dot(vec a, vec b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

vec cross(vec a, vec b) {
    return vec({
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    });
}

double norm2(vec point) {
    return point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
}

double norm(vec point) {
    return std::sqrt(norm2(point));
}


double dist2(vec point1, vec point2) {
    return norm2(point2 - point1);
}

double dist(vec point1, vec point2) {
    return norm(point2 - point1);
}

int sign(double x) {
    return std::signbit(x) ? -1 : 1;
}

vec delta_pbc(vec point1, vec point2, vec box) {
    vec delta = point2 - point1;
    delta[0] = abs(delta[0]) * 2 > box[0] ? delta[0] - sign(delta[0]) * box[0] : delta[0];
    delta[1] = abs(delta[1]) * 2 > box[1] ? delta[1] - sign(delta[1]) * box[1] : delta[1];
    delta[2] = abs(delta[2]) * 2 > box[2] ? delta[2] - sign(delta[2]) * box[2] : delta[2];

    return delta;
}

// Test speed
double dist2_pbc(vec point1, vec point2, vec box) {
    return norm2(delta_pbc(point1, point2, box));
}

double dist_pbc(vec point1, vec point2, vec box) {
    return norm(delta_pbc(point1, point2, box));
}

vec rotate_point(vec point, double psi, double theta, double phi) {
    double cos_psi = std::cos(psi);
    double sin_psi = std::sin(psi);

    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    double cos_phi = std::cos(phi);
    double sin_phi = std::sin(phi);

    // Formula from http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    return vec({
        cos_theta * cos_phi * point[0] + (sin_psi * sin_theta * cos_phi - cos_psi * sin_phi) * point[1] + (cos_psi * sin_theta * cos_phi + sin_psi * sin_phi) * point[2],
        cos_theta * sin_phi * point[0] + (sin_psi * sin_theta * sin_phi + cos_psi * cos_phi) * point[1] + (cos_psi * sin_theta * sin_phi - sin_psi * cos_phi) * point[2],
        -sin_theta * point[0] + sin_psi * cos_theta * point[1] + cos_psi * cos_theta * point[2]
    });
}
