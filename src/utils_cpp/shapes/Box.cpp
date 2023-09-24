#include "Box.hpp"

Box::Box() {}
Box::Box(vec center_, vec borders_) : Shape(center_), borders(borders_) {}

Box::Box(const Box& lhs) : Shape(lhs), borders(lhs.borders) {}

Box& Box::operator=(const Box& lhs) {
    Box t(lhs);
    std::swap(center, t.center);
    std::swap(borders, t.borders);

    return *this;
}

Box::Box(Box&& rhs) : Shape(rhs), borders(rhs.borders)  {
    rhs.center = vec();
    rhs.borders = vec();
}

Box& Box::operator=(Box&& rhs) {
    Box t(std::move(rhs));
    std::swap(center, t.center);
    std::swap(borders, t.borders);

    return *this;
}


double Box::get_volume() const {
    return borders[0] * borders[1] * borders[2];
}

double Box::get_surface() const {
    return 2 * (borders[0] * borders[1] + borders[1] * borders[2] + borders[0] * borders[2]);
}

bool Box::check_point(vec point) const {
    vec shifted_pos = point - center;
    return (abs(shifted_pos[0]) < borders[0] / 2) &&
           (abs(shifted_pos[1]) < borders[1] / 2) &&
           (abs(shifted_pos[2]) < borders[2] / 2);
}

vec Box::generate_point(std::mt19937& gen) const {
    std::uniform_real_distribution<double> dist_x(0, borders[0]);
    std::uniform_real_distribution<double> dist_y(0, borders[1]);
    std::uniform_real_distribution<double> dist_z(0, borders[2]);

    return vec({dist_x(gen), dist_y(gen), dist_z(gen)});
}
