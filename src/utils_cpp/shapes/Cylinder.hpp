#include <random>
#include <cmath>

#include "Shape.hpp"
#include "../alg/pbc.hpp"
#include "../handler.hpp"

#ifndef CYLINDER
#define CYLINDER

class Cylinder : public Shape {
public:
    double radius;
    double length;
    vec axis;

    // RAII
    Cylinder();
    Cylinder(vec center_, double radius_, double length_, vec axis_);
    Cylinder(const Cylinder& lhs);
    Cylinder& operator=(const Cylinder& lhs);
    Cylinder(Cylinder&& rhs);
    Cylinder& operator=(Cylinder&& rhs);

    // Methods
    double get_volume() const override;
    double get_surface() const override;
    bool check_point(vec point) const override;
    vec generate_point(std::mt19937& gen) const override;
};

#endif
