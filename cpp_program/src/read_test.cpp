#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <iostream>
#include <fstream>

// Usiamo lo namespace deal.II
using namespace dealii;

// Programma minimale per testare la lettura di un file VTU.
int main(int argc, char *argv[])
{
  // Controlla che sia stato fornito un nome di file
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <path_to_vtu_file>" << std::endl;
      return 1;
    }

  const std::string vtu_filename = argv[1];
  std::cout << "Attempting to read mesh from: " << vtu_filename << std::endl;

  try
    {
      // Crea gli oggetti necessari
      Triangulation<3> mesh;
      GridIn<3>        grid_in;
      grid_in.attach_triangulation(mesh);

      // Prova ad aprire il file
      std::ifstream input_file(vtu_filename);
      if (!input_file)
        {
          std::cerr << "Error: Cannot open file!" << std::endl;
          return 1;
        }

      // Prova a leggere il file VTU
      grid_in.read_vtu(input_file);

      // Se arriva qui, la lettura ha avuto successo
      std::cout << "\nSUCCESS! Mesh read successfully." << std::endl;
      std::cout << "Number of active cells: " << mesh.n_active_cells() << std::endl;
    }
  catch (const std::exception &e)
    {
      // Se c'è un'eccezione, stampala. Questo ci darà più dettagli.
      std::cerr << "\nERROR! An exception was caught:" << std::endl;
      std::cerr << "  " << e.what() << std::endl;
      return 1;
    }

  return 0;
}