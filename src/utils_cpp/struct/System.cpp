#include "System.hpp"

System::System() {}

System::System(std::string title_, vec box_) : title(title_), box(box_) {}

System::System(const System& lhs) : title(lhs.title), box(lhs.box), atoms(lhs.atoms), points(lhs.points) {}

System& System::operator=(const System& lhs) {
    System t(lhs);
    std::swap(title, t.title);
    std::swap(box, t.box);
    std::swap(atoms, t.atoms);
    std::swap(points, t.points);

    return *this;
}

System::System(System&& rhs) : title(rhs.title), box(rhs.box), atoms(rhs.atoms), points(rhs.points) {
    rhs.title = "System";
    rhs.box = vec();
    rhs.atoms.clear();
    rhs.points.clear();
}

System& System::operator=(System&& rhs) {
    System t(std::move(rhs));
    std::swap(title, t.title);
    std::swap(box, t.box);
    std::swap(atoms, t.atoms);
    std::swap(points, t.points);

    return *this;
}


void System::add_mol(const Molecule& mol_, size_t mol_id_) {
    for (Atom a : mol_.atoms) {
        a.mol_id = mol_id_;
        a.id = atoms.size() + 1;
        atoms.push_back(a);
    }
}

void System::apply_pbc() {
    vec half_box = box / 2;
    for (Atom& a : atoms) {
        a.xyz[0] = abs(a.xyz[0] - half_box[0]) > half_box[0] ? a.xyz[0] - box[0] * sign(a.xyz[0]) : a.xyz[0];
        a.xyz[1] = abs(a.xyz[1] - half_box[1]) > half_box[1] ? a.xyz[1] - box[1] * sign(a.xyz[1]) : a.xyz[1];
        a.xyz[2] = abs(a.xyz[2] - half_box[2]) > half_box[2] ? a.xyz[2] - box[2] * sign(a.xyz[2]) : a.xyz[2];
    }
}

vec System::get_center() {
    return box / 2;
}
