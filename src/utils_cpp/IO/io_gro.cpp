#include "io_gro.hpp"

System read_sys(std::filesystem::path file_path) {
    std::ifstream file(file_path);
    System sys;
    Atom atom;

    std::string line, subline, word;
    std::vector<std::string> args;
    std::array<int, 9> mask = {5, 5, 5, 5, 8, 8, 8, 100};
    size_t point = 0;

    // System title
    std::getline(file, line);
    line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
    sys.title = line;

    // Number of atoms
    std::getline(file, line);
    size_t size = std::stoul(line);

    for (size_t i = 0; i < size; i++) {
        std::getline(file, line);
        for (size_t i = 0; i < 7; i++) {
            subline = std::string(line.substr(point, mask[i]));
            subline.erase(std::remove_if(subline.begin(), subline.end(), isspace), subline.end());
            args.push_back(subline);
            point += mask[i];
        }

        atom.mol_id = std::stoi(args[0]);
        atom.mol_name = args[1];
        atom.name = args[2];
        atom.id = std::stoi(args[3]);
        atom.xyz = vec({std::stod(args[4]), std::stod(args[5]), std::stod(args[6])});

        sys.atoms.push_back(atom);
        args.clear();
        point = 0;
    }

    // Box
    for (std::string line; std::getline(file, line, ' '); ) {
        if (!line.empty()) {
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            args.push_back(line);
        }
    }
    sys.box = vec({std::stod(args[0]), std::stod(args[1]), std::stod(args[2])});

    return sys;

}


Molecule read_mol(std::filesystem::path file_path) {
    System sys = read_sys(file_path);
    Molecule mol;

    mol.id = sys.atoms[0].mol_id;
    mol.name = sys.atoms[0].mol_name;

    for (Atom& a : sys.atoms) {
        if (mol.id != a.mol_id || mol.name != a.mol_name) {
            std::cerr << "The file contains several molecules, and should contain one." << std::endl;
        }

        mol.atoms.push_back(a);
    }

    mol.set_atoms_to_center();
    mol.calc_size();

    return mol;

}


void write_sys(const System& sys, std::filesystem::path file_path) {
    std::ofstream file(file_path);

    file << sys.title << std::endl;
    file << sys.atoms.size() << std::endl;

    for (const Atom& a : sys.atoms) {
        file << std::setw(5) << a.mol_id << std::setw(5) << a.mol_name << std::setw(5) << a.name << std::setw(5) << a.id;
        file << a.xyz << std::endl;
    }

    file << sys.box << std::endl;
}
