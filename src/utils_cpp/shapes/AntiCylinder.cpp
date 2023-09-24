#include "AntiCylinder.hpp"

AntiCylinder::AntiCylinder() {}

AntiCylinder::AntiCylinder(vec center_, double radius_, double length_, vec axis_, vec borders_center_, vec borders_) : Shape(center_) {
    cylinder = Cylinder(center_, radius_, length_, axis_);
    box = Box(borders_center_, borders_);
}

AntiCylinder::AntiCylinder(const Cylinder& cylinder_, const Box& box_) : Shape(cylinder_.center) {
    cylinder = cylinder_;
    box = box_;
}

AntiCylinder::AntiCylinder(const AntiCylinder& lhs) : Shape(lhs) {
    cylinder = lhs.cylinder;
    box = lhs.box;
}

AntiCylinder& AntiCylinder::operator=(const AntiCylinder& lhs) {
    AntiCylinder t(lhs);
    std::swap(cylinder, t.cylinder);
    std::swap(box, t.box);

    return *this;
}

AntiCylinder::AntiCylinder(AntiCylinder&& rhs) : Shape(rhs) {
    cylinder = rhs.cylinder;
    box = rhs.box;

    rhs.box = Box();
    rhs.cylinder = Cylinder();
}

AntiCylinder& AntiCylinder::operator=(AntiCylinder&& rhs) {
    AntiCylinder t(std::move(rhs));
    std::swap(cylinder, t.cylinder);
    std::swap(box, t.box);

    return *this;
}


double AntiCylinder::get_volume() const {
    return box.get_volume() - cylinder.get_volume();
}

double AntiCylinder::get_surface() const {
    return box.get_surface() + cylinder.get_surface();
}

bool AntiCylinder::check_point(vec point) const {
    return box.check_point(point) && !cylinder.check_point(point);
}

vec AntiCylinder::generate_point(std::mt19937& gen) const {
    bool inside_cylinder = true;
    vec point;
    while (inside_cylinder) {
        point = box.generate_point(gen);
        inside_cylinder = cylinder.check_point(point);
    }

    return point;
}
