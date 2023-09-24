#include <random>

#include "../struct/Atom.hpp"
#include "../struct/Molecule.hpp"
#include "../struct/System.hpp"
#include "../shapes/Shape.hpp"
#include "../shapes/Box.hpp"
#include "../timer.hpp"
#include "pbc.hpp"
#include <cmath>

#include "../handler.hpp"

bool check_points_intersections(System& sys, vec point, double min_dist2);
bool check_mol_intersections(System& sys, const Molecule& mol, vec point, double min_dist2);

void insert_mol_into_shape(
    System& sys,
    const Shape& shape,
    Molecule mol,
    size_t mol_id,
    std::mt19937& gen,
    double min_dist2 = 0.0064,
    double package = 0.4,
    int insertion_limit = int(1e5),
    int rotation_limit = 10
);

void push_atoms_apart(
    System& sys,
    std::mt19937& gen,
    double min_dist2 = 0.0064,
    double max_dist2 = 0.0144,
    int iteration_lim = 10
);

void mol_rotation(Molecule& mol, double psi, double theta, double phi);
void random_mol_rotation(Molecule& mol, std::mt19937& gen);
