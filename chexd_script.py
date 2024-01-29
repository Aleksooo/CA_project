import numpy as np
import argparse
import os
from src.utils_py.io.gro import read_gro, write_gro
from src.utils_py.geom.Box import Box
from src.utils_py.geom.Cylinder import Cylinder
from src.utils_py.assembler.build import build_system

parser = argparse.ArgumentParser()
parser.add_argument('H', type=float, help='Slot H')
parser.add_argument('phi', type=float, help='Amoung of water')
parser.add_argument('build', type=int, help='Build system flag')
parser.add_argument('path', type=str, help='path on server')
parser.add_argument('folder', type=str, help='folder on server and local')
# parser.add_argument('sc', type=float, help='scale factor')

args = parser.parse_args()
structure = read_gro(f'chexd/chexd.gro')

WIDTH_X, WIDTH_Y, HEIGHT = structure.box

H = args.H
# H = 20

offset = 2
delta_h = 0.2
frac = args.phi

# Конфигурация половины
folder = args.folder
ff = 'trappe'
dir = 'ff/'+ff

''' ПАРАМЕТРЫ НЕ ТРОГАТЬ!!! '''
insertion_limit = int(1e5)
rotation_limit = 1000
package = 0.3
distance = {'min': 0.08**2, 'opt': 0.12**2}

system_size = np.array([WIDTH_X, WIDTH_Y, HEIGHT + H])
points = structure.atoms_xyz
structure.box = system_size

names = ['tip4p']
density = [33.3277] # nm-3


# XZ surface
# radius = np.sqrt(frac * WIDTH_X * H / np.pi)
# cylinder = Cylinder(
#     center=np.array([WIDTH_X/2, WIDTH_Y/2, HEIGHT + radius + delta_h]),
#     radius=radius,
#     length=WIDTH_Y,
#     axis=np.array([0, 1, 0])
# )

box_left = Box(
    center=np.array([WIDTH_X*(1-frac)/2, WIDTH_Y/2, HEIGHT + H/2]),
    borders=np.array([WIDTH_X*(1-frac), WIDTH_Y, H])
)

box_left_delta = Box(
    center=np.array([WIDTH_X*(1-frac)/2, WIDTH_Y/2, HEIGHT + H/2]),
    borders=np.array([WIDTH_X*(1-frac), WIDTH_Y, H - 2 * delta_h])
)


insert_shapes = [box_left_delta]
shapes = [box_left]
numbers = list(np.round(np.array([shapes[i].get_volume() * density[i] for i in range(len(names))])).astype(int))

if args.build:
    structure = build_system(
        dir, structure, names, numbers, insert_shapes, points,
        insertion_limit=insertion_limit,
        rotation_limit=rotation_limit,
        package=package,
        min_dist2=distance['min']
    )


# Запись .gro и system.itp
mol_names = ''.join(map(lambda x: 'w' if x in ['spce', 'tip4p'] else x[0], names))
mol_nums = '_'.join(map(str, numbers))
filename = 'chexd_'+ mol_names + '_' + mol_nums

os.system(f'mkdir systems/{folder}')

with open(f'systems/{folder}/system.itp', 'w') as f:
    for name in names:
        f.write(f'#include "{name}.itp"\n')
    f.write('#include "chexd.itp"\n')

    f.write(f'\n[ system ]\n{filename}\n')
    f.write('\n[ molecules ]\n; molecule name\tnr.\n')
    f.write('#include "chexdmol.itp"\n')
    for i, name in enumerate(names):
        f.write(f'{name}\t{numbers[i]}\n')

print('Writing .gro files.')

if args.build:
    with open(f'systems/{folder}/{filename}.gro', 'w') as f:
        f.write(write_gro(structure))

    # writing backup
    with open(f'systems/{folder}/#{filename}.gro#', 'w') as f:
        f.write(write_gro(structure))

    # Перемешивание
    print('Mixing system')
    os.system(f'./mixer -f systems/{folder}/{filename}.gro -o systems/{folder}/{filename}.gro -mn2 {distance["min"]} -opt2 {distance["opt"]}')

# Создание скрипта для запуска
# for sc in [1.5, 2, 4]:
#     with open(f'systems/{folder}/run_{sc}.sh', 'w') as f:
#         f.write(f'#!/bin/bash\n#SBATCH --job-name=gromacs_{filename}\n#SBATCH --partition=RT\n#SBATCH --nodes=3\n#SBATCH --ntasks-per-node=16\n\n')
#         f.write(f'srun -n 1 gmx_mpi grompp -f ../nvt_cal_steep.mdp -c {filename}_init.gro -p trappe_{sc}.top -o {filename} -maxwarn 10\n')
#         f.write(f'srun -n 16 gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename}\n')
#         f.write(f'rm ./#*#\n')
#         f.write(f'rm *pdb\n')
#         f.write(f'srun -n 1 gmx_mpi grompp -f ../nvt_cal_short.mdp -c {filename}.gro -p trappe_{sc}.top -o {filename} -maxwarn 10\n')
#         f.write(f'srun -n 48 gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename} -dlb yes\n')
#         f.write(f'rm ./#*#\n')
#         f.write(f'srun -n 1 gmx_mpi grompp -f ../nvt_cal_run.mdp -c {filename}.gro -p trappe_{sc}.top -o {filename} -maxwarn 10\n')
#         f.write(f'srun -n 48 gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename} -dlb yes\n')
#         f.write(f'rm ./#*#')

server_name = 'softcluster'

with open(f'systems/{folder}/run.sh', 'w') as f:
    f.write(f'''#!/bin/bash
#SBATCH -J gromacs
#SBATCH -N 1\t# Number of nodes requested
#SBATCH -n 16\t# Total number of mpi tasks requested
#SBATCH --cpus-per-task 2\t# Total number of omp tasks requested

gmx grompp -f nvt_chexd_steep.mdp -c {filename}_init.gro -p {ff}.top -o {filename} -maxwarn 10
prun gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename} -ntomp 2 -gpu_id 0
rm ./#*#
rm *pdb

gmx grompp -f nvt_chexd_short.mdp -c {filename}.gro -p {ff}.top -o {filename} -maxwarn 10
prun gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename} -ntomp 2 -gpu_id 0
rm ./#*#

gmx grompp -f nvt_chexd_run.mdp -c {filename}.gro -p {ff}.top -o {filename} -maxwarn 10
prun gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename} -ntomp 2 -gpu_id 0
rm ./#*#
    ''')

#     f.write(f'''#!/bin/bash
# #SBATCH --job-name=gromacs
# #SBATCH --partition=RT
# #SBATCH -N 1
# #SBATCH -n 16

# srun -n 1 gmx_mpi grompp -f nvt_cal_steep.mdp -c {filename}_init.gro -p trappe.top -o {filename} -maxwarn 10
# srun gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename}
# rm ./#*#
# rm *pdb

# srun -n 1 gmx_mpi grompp -f nvt_cal_short.mdp -c {filename}.gro -p trappe.top -o {filename} -maxwarn 10
# srun gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename} -dlb yes
# rm ./#*#

# srun -n 1 gmx_mpi grompp -f nvt_cal_run.mdp -c {filename}.gro -p trappe.top -o {filename} -maxwarn 10
# srun gmx_mpi mdrun -s -o -x -c -e -g -v -deffnm {filename} -dlb yes
# rm ./#*#''')


# Отправка файлов на сервер при перезапуске
server_name = 'softcluster'
# server_name = 'mipt-cluster'

files = list((args.path+'/'+folder).split('/'))
for i in range(len(files)):
    os.system(f"ssh {server_name} 'mkdir {'~/'+'/'.join(files[:i+1])}'")

files = ['tip4p.itp', 'chexd.itp']
for file in files:
    os.system(f'scp {dir}/itp/{file} {server_name}:{args.path}/{folder}')
os.system(f'scp chexd/chexdmol.itp {server_name}:{args.path}/{folder}')

os.system(f'scp {dir}/{ff}.top {server_name}:{args.path}/{folder}/{ff}.top')

files = ['nvt_chexd_steep.mdp', 'nvt_chexd_short.mdp', 'nvt_chexd_run.mdp']
for file in files:
    os.system(f'scp {dir}/mdp/{file} {server_name}:{args.path}/{folder}')

# files = [filename+'.gro', 'system.itp', 'run_1.5.sh', 'run_2.sh', 'run_4.sh']
files = [filename+'.gro', 'system.itp', 'run.sh']
for file in files:
    os.system(f'scp systems/{folder}/{file} {server_name}:{args.path}/{folder}')
os.system(f'scp systems/{folder}/{filename}.gro {server_name}:{args.path}/{folder}/{filename}_init.gro')
