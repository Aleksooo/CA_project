#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <array>
#include <vector>
#include "../struct/Atom.hpp"
#include "../struct/Molecule.hpp"
#include "../struct/System.hpp"

#include "../handler.hpp"

#ifndef IO
#define IO

System read_sys(std::filesystem::path file_path);
Molecule read_mol(std::filesystem::path file_path);
void write_sys(const System& sys, std::filesystem::path file_path);

#endif
