### Materiali utili 
Adaptive space: https://www.dealii.org/current/doxygen/deal.II/step_6.html
Adaptive time: https://www.sciencedirect.com/science/article/pii/S0045782598001212 

### Organizing the source code
Please place all your sources into the `src` folder.

Binary files must not be uploaded to the repository (including executables).

Mesh files should not be uploaded to the repository. If applicable, upload `gmsh` scripts with suitable instructions to generate the meshes (and ideally a Makefile that runs those instructions). If not applicable, consider uploading the meshes to a different file sharing service, and providing a download link as part of the building and running instructions.

### Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```
The executable will be created into `build`, and can be executed through
```bash
$ ./executable-name
```
