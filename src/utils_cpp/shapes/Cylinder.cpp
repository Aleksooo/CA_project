#include "Cylinder.hpp"

Cylinder::Cylinder() {}
Cylinder::Cylinder(vec center_, double radius_, double length_, vec axis_) : Shape(center_), radius(radius_), length(length_), axis(axis_ / norm(axis_)) {}

Cylinder::Cylinder(const Cylinder& lhs) : Shape(lhs), radius(lhs.radius), length(lhs.length), axis(lhs.axis) {}

Cylinder& Cylinder::operator=(const Cylinder& lhs) {
    Cylinder t(lhs);
    std::swap(center, t.center);
    std::swap(radius, t.radius);
    std::swap(length, t.length);
    std::swap(axis, t.axis);

    return *this;
}

Cylinder::Cylinder(Cylinder&& rhs) : Shape(rhs), radius(rhs.radius), length(rhs.length), axis(rhs.axis)  {
    rhs.center = vec();
    rhs.radius = 0;
    rhs.length = 0;
    rhs.axis = vec();
}

Cylinder& Cylinder::operator=(Cylinder&& rhs) {
    Cylinder t(std::move(rhs));
    std::swap(center, t.center);
    std::swap(radius, t.radius);
    std::swap(length, t.length);
    std::swap(axis, t.axis);

    return *this;
}


double Cylinder::get_volume() const {
    return M_PI * radius * radius * length;
}

double Cylinder::get_surface() const {
    return 2 * M_PI * radius * (radius + length);
}

bool Cylinder::check_point(vec point) const {
    double d = norm(cross(point - center, axis));
    double l = abs(dot(point - center, axis));

    return (d < radius) && (2 * l < length);
}

// vec Cylinder::generate_point(std::mt19937& gen) const {
//     std::uniform_real_distribution<double> dist_theta(0, M_PI);
//     std::uniform_real_distribution<double> dist_phi(0, 2 * M_PI);
//     double theta = dist_theta(gen);
//     double phi = dist_phi(gen);

//     std::uniform_real_distribution<double> dist_r(0, radius);
//     double r = dist_r(gen);
//     vec sphere_vec({
//         std::sin(theta) * std::cos(phi),
//         std::sin(theta) * std::sin(phi),
//         std::cos(theta)
//     });
//     vec r_vec = sphere_vec - axis * dot(sphere_vec, axis);
//     r_vec = r_vec * r / norm(r_vec);

//     std::uniform_real_distribution<double> dist_z(-length / 2, length / 2);
//     double z = dist_z(gen);

//     return center + axis * z + r_vec;
// }

vec Cylinder::generate_point(std::mt19937& gen) const {
    std::uniform_real_distribution<double> dist_r(0, 1);
    std::uniform_real_distribution<double> dist_phi(0, 2 * M_PI);
    std::uniform_real_distribution<double> dist_z(-length / 2, length / 2);

    double r = radius * std::sqrt(dist_r(gen));
    double phi = dist_phi(gen);
    double z = dist_z(gen);

    vec pos({r * std::cos(phi), r * std::sin(phi), z});
    pos = rotate_point(pos, 0, std::acos(axis[2]), 0);

    return center + pos;
}
