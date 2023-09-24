#include <random>
#include <cmath>

#include "Shape.hpp"
#include "Box.hpp"
#include "Cylinder.hpp"
#include "../alg/pbc.hpp"
#include "../handler.hpp"

#ifndef ANTICYLINDER
#define ANTICYLINDER

class AntiCylinder : public Shape {
public:
    Cylinder cylinder;
    Box box;

    // RAII
    AntiCylinder();
    AntiCylinder(vec center_, double radius_, double length_, vec axis_, vec borders_center_, vec borders_);
    AntiCylinder(const Cylinder& cylinder_, const Box& box_);
    AntiCylinder(const AntiCylinder& lhs);
    AntiCylinder& operator=(const AntiCylinder& lhs);
    AntiCylinder(AntiCylinder&& rhs);
    AntiCylinder& operator=(AntiCylinder&& rhs);

    // Methods
    double get_volume() const override;
    double get_surface() const override;
    bool check_point(vec point) const override;
    vec generate_point(std::mt19937& gen) const override;
};

#endif
