#include <iostream>
#include <vector>
#include <numeric>
#include "Atom.hpp"
#include "../alg/pbc.hpp"

#include "../handler.hpp"

#ifndef MOLECULE
#define MOLECULE

class Molecule {
public:
    int id = 1;
    std::string name = "MOL";
    std::vector<Atom> atoms;
    double size;

    // RAII
    Molecule();
    Molecule(const Molecule& lhs);
    Molecule& operator=(const Molecule& lhs);
    Molecule(Molecule&& rhs);
    Molecule& operator=(Molecule&& rhs);

    // Methods
    vec get_center();
    void set_atoms_to_center();
    void set_atoms_to_point(vec point);
    void calc_size();
};

#endif
