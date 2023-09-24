#include "Atom.hpp"

Atom::Atom() {}
Atom::Atom(const Atom& lhs) : mol_id(lhs.mol_id), mol_name(lhs.mol_name), id(lhs.id), name(lhs.name), xyz(lhs.xyz) {}

Atom& Atom::operator=(const Atom& lhs) {
    Atom t(lhs);
    std::swap(mol_id, t.mol_id);
    std::swap(mol_name, t.mol_name);
    std::swap(id, t.id);
    std::swap(name, t.name);
    std::swap(xyz, t.xyz);

    return *this;
}

Atom::Atom(Atom&& rhs) : mol_id(rhs.mol_id), mol_name(rhs.mol_name), id(rhs.id), name(rhs.name), xyz(rhs.xyz) {
    rhs.mol_id = 1;
    rhs.mol_name = "MOL";
    rhs.id = 1;
    rhs.name = "ATOM";
    rhs.xyz = vec();
}

Atom& Atom::operator=(Atom&& rhs) {
    Atom t(std::move(rhs));
    std::swap(mol_id, t.mol_id);
    std::swap(mol_name, t.mol_name);
    std::swap(id, t.id);
    std::swap(name, t.name);
    std::swap(xyz, t.xyz);

    return *this;
}
