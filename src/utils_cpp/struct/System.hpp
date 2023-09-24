#include <iostream>
#include <vector>
#include "Atom.hpp"
#include "Molecule.hpp"

#include "../handler.hpp"

#ifndef SYSTEM
#define SYSTEM

class System {
public:
    std::string title = "System";
    vec box;
    std::vector<Atom> atoms;
    std::vector<vec> points;

    // RAII
    System();
    System(std::string title_, vec box_);
    System(const System& lhs);
    System& operator=(const System& lhs);
    System(System&& rhs);
    System& operator=(System&& rhs);

    // Methods
    void add_mol(const Molecule& mol_, size_t mol_id_);
    void apply_pbc();
    vec get_center();
};

#endif
