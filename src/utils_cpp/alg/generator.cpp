#include "generator.hpp"

bool check_points_intersections(System& sys, vec point, double min_dist2) {
    for (const vec& p : sys.points) {
        if (dist2_pbc(p, point, sys.box) < min_dist2) {
        // if (dist2(p, point) < min_dist2) {
            return true;
        }
    }

    return false;
}

bool check_mol_intersections(System& sys, const Molecule& mol, vec point, double min_dist2) {
    for (const Atom& a1 : mol.atoms) {
        for (const Atom& a2 : sys.atoms) {
            if (dist2_pbc(a1.xyz + point, a2.xyz, sys.box) < min_dist2) {
            // if (dist2(a1.xyz + point, a2.xyz) < min_dist2) {
                return true;
            }
        }
    }

    return false;
}

void insert_mol_into_shape(
    System& sys,
    const Shape& shape,
    Molecule mol,
    size_t mol_id,
    std::mt19937& gen,
    double min_dist2,
    double package,
    int insertion_limit,
    int rotation_limit
) {
    bool overlap = true;
    int insertion_counter = 0;
    int rotation_counter = 0;

    vec new_point;

    // Try to insert point
    while (overlap && (insertion_counter < insertion_limit)) {
        insertion_counter++;

        new_point = shape.generate_point(gen);
        overlap = check_points_intersections(
            sys,
            new_point,
            (2 - insertion_counter / insertion_limit) * package * mol.size
            // 2 * package * mol.size
        );
    }
    sys.points.push_back(new_point);

    overlap = true;
    // Rotating mol around point
    while (overlap && (rotation_counter < rotation_limit)) {
        rotation_counter++;

        random_mol_rotation(mol, gen);
        overlap = check_mol_intersections(sys, mol, new_point, min_dist2);
    }

    if (insertion_counter == insertion_limit) { std::cerr << "Can't pos mol #" << mol_id << std::endl; }

    mol.set_atoms_to_point(new_point);
    sys.add_mol(mol, mol_id);
}


void push_atoms_apart(
    System& sys,
    std::mt19937& gen,
    double min_dist2,
    double opt_dist2,
    int iteration_lim
) {
    double min_dist = sqrt(min_dist2);
    double opt_dist = sqrt(opt_dist2);
    std::uniform_real_distribution<double> dist(min_dist, opt_dist);

    int iter = 0;
    bool overlap = true;
    int overlap_counter = 0;

    Timer<s> timer;

    while (overlap && (iter < iteration_lim)) {
        iter++;
        overlap = false;
        overlap_counter = 0;

        std::cout << "Iteration " << iter;
        timer.start();

        for (size_t i = 0; i < sys.atoms.size() - 1; i++) {
            // if (sys.atoms[i].name[0] == 'H') {
            //     continue;
            // }

            for (size_t j = i + 1; j < sys.atoms.size(); j++) {
                // if ((sys.atoms[i].mol_id != sys.atoms[j].mol_id) && (sys.atoms[j].name[0] != 'H')) {
                if (sys.atoms[i].mol_id != sys.atoms[j].mol_id) {
                    vec delta = delta_pbc(sys.atoms[i].xyz, sys.atoms[j].xyz, sys.box);

                    if (norm(delta) < min_dist) {
                        sys.atoms[j].xyz = sys.atoms[j].xyz + delta * dist(gen) / norm(delta);
                        overlap = true;
                        overlap_counter++;
                    }
                }
            }
        }
        timer.stop();
        std::cout << " - " << std::setprecision(3) << timer.get_time() << "s" << std::endl;
        timer.reset();

        std::cout << overlap_counter << " overlaps detected" << std::endl;
    }

    sys.apply_pbc();
}


void mol_rotation(Molecule& mol, double psi, double theta, double phi) {
    for (Atom& a : mol.atoms) {
        a.xyz = rotate_point(a.xyz, psi, theta, phi);
    }
}

void random_mol_rotation(Molecule& mol, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(0, 2 * M_PI);
    mol_rotation(mol, dist(gen), dist(gen), dist(gen));
}
