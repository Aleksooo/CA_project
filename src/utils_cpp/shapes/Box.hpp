#include <random>

#include "Shape.hpp"
#include "../handler.hpp"

#ifndef BOX
#define BOX

class Box : public Shape {
public:
    vec borders;

    // RAII
    Box();
    Box(vec center_, vec borders_);
    Box(const Box& lhs);
    Box& operator=(const Box& lhs);
    Box(Box&& rhs);
    Box& operator=(Box&& rhs);

    // Methods
    double get_volume() const override;
    double get_surface() const override;
    bool check_point(vec point) const override;
    vec generate_point(std::mt19937& gen) const override;
};

#endif
