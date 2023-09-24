#include <iostream>
#include <vector>

#include "../handler.hpp"

#ifndef ATOM
#define ATOM

class Atom {
public:
    int mol_id = 1;
    std::string mol_name = "MOL";
    int id = 1;
    std::string name = "ATOM";
    vec xyz;

    // RAII
    Atom();
    Atom(const Atom& lhs);
    Atom& operator=(const Atom& lhs);
    Atom(Atom&& rhs);
    Atom& operator=(Atom&& rhs);


};

#endif
