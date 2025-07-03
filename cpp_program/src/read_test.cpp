#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <iostream>
#include <fstream>

using namespace dealii;

// Minimal program to test reading a VTU file.
int main(int argc, char *argv[])
{
  // Check that a filename was provided
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <path_to_vtu_file>" << std::endl;
      return 1;
    }

  const std::string vtu_filename = argv[1];
  std::cout << "Attempting to read mesh from: " << vtu_filename << std::endl;

  try
    {
      // Create the necessary objects
      Triangulation<3> mesh;
      GridIn<3>        grid_in;
      grid_in.attach_triangulation(mesh);

      std::ifstream input_file(vtu_filename);
      if (!input_file)
        {
          std::cerr << "Error: Cannot open file!" << std::endl;
          return 1;
        }

      
      grid_in.read_vtu(input_file);

      // Correctly read the mesh
      std::cout << "\nSUCCESS! Mesh read successfully." << std::endl;
      std::cout << "Number of active cells: " << mesh.n_active_cells() << std::endl;
    }
  catch (const std::exception &e)
    {
      // Handle any exceptions that may occur
      std::cerr << "\nERROR! An exception was caught:" << std::endl;
      std::cerr << "  " << e.what() << std::endl;
      return 1;
    }

  return 0;
}