#include <iostream>
#include <iomanip>
#include <filesystem>
#include "handler.hpp"
#include "struct/Atom.hpp"
#include "struct/Molecule.hpp"
#include "struct/System.hpp"
#include "IO/io_gro.hpp"
#include "shapes/Box.hpp"
#include "shapes/Cylinder.hpp"
#include "shapes/AntiCylinder.hpp"
#include "alg/generator.hpp"
#include "random"

#include "timer.hpp"

const double MIN_DIST2 = 0.08 * 0.08;
const double MAX_DIST2= 0.12 * 0.12;
const double PACKAGE = 0.4;

// const int DECANE_NUM = 2340;
// const int WATER_NUM = 111133;
const int DECANE_NUM = 293;
const int WATER_NUM = 13892;

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());
    Timer<s> timer;

    // System sys("Decane in water", vec({18, 18, 12}));
    System sys("Decane in water", vec({9, 9, 6}));

    // Decane
    std::filesystem::path decane_inp("../ff/gromos/gro/decane.gro");
    Molecule decane = read_mol(decane_inp);
    double radius = 1.75;
    // double radius = 1;
    double length = sys.box[0];
    Cylinder cylinder(vec({sys.box[0] / 2, sys.box[1] / 2, radius}), radius, length, vec({1, 0, 0}));

    std::cout << "Inserting decane with density " << std::setprecision(3) << DECANE_NUM / cylinder.get_volume() << std::endl;
    timer.start();
    for (size_t mol_id = 1; mol_id <= DECANE_NUM; mol_id++) {
        std::cout << mol_id << std::endl;
        insert_mol_into_shape(sys, cylinder, decane, mol_id, gen, MIN_DIST2, PACKAGE);
    }
    timer.stop();
    std::cout << "Insertation time: " << timer.get_time() << ' ' << get_name(ms) << '\n' << std::endl;


    // Water
    std::filesystem::path water_inp("../ff/gromos/gro/water.gro");
    Molecule water = read_mol(water_inp);
    AntiCylinder anticylinder(cylinder, Box(sys.get_center(), sys.box));

    std::cout << "Inserting water with density " << std::setprecision(3) << WATER_NUM / anticylinder.get_volume() << std::endl;
    timer.reset();
    timer.start();
    for (size_t mol_id = 1; mol_id <= WATER_NUM; mol_id++) {
        std::cout << mol_id << std::endl;
        insert_mol_into_shape(sys, anticylinder, water, mol_id, gen);
    }
    timer.stop();
    std::cout << "Insertation time: " << timer.get_time() << ' ' << get_name(ms) << '\n' << std::endl;

    std::cout << "Pushing apart..." << std::endl;
    timer.reset();
    timer.start();
    push_atoms_apart(sys, gen, MIN_DIST2, MAX_DIST2);
    timer.stop();
    std::cout << "Pushing time: " << timer.get_time() << ' ' << get_name(ms) << '\n' << std::endl;

    std::filesystem::path outp("../decane_water_article.gro");
    write_sys(sys, outp);

    return 0;
}
