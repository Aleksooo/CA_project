#include "Shape.hpp"

Shape::Shape() {}
Shape::Shape(vec center_) : center(center_) {}

Shape::Shape(const Shape& lhs) : center(lhs.center) {}

// Shape& Shape::operator=(const Shape& lhs) {}

Shape::Shape(Shape&& rhs) : center(rhs.center){
    rhs.center = vec();
}

// Shape& Shape::operator=(Shape&& rhs) {}
