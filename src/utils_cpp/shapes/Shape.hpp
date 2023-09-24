#include <random>
#include "../handler.hpp"

#ifndef SHAPE
#define SHAPE

class Shape {
public:
    vec center;

    // RAII
    Shape();
    Shape(vec center_);
    Shape(const Shape& lhs);
    // Shape& operator=(const Shape& lhs);
    Shape(Shape&& rhs);
    // Shape& operator=(Shape&& rhs);

    // Methods
    virtual double get_volume() const = 0;
    virtual double get_surface() const = 0;
    virtual bool check_point(vec point) const = 0;
    virtual vec generate_point(std::mt19937& gen) const = 0;
};

#endif
