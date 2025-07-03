#include "Heat.hpp"
#include <iostream> // Aggiunto per std::cerr
#include <string>   // Aggiunto per std::stoul e std::stod

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  // Controlla che siano stati forniti i due argomenti necessari
  if (argc != 3)
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cerr << "Usage: " << argv[0] << " <refinement_level> <deltat>"
                    << std::endl;
        }
      return 1; // Termina con errore se gli argomenti mancano
    }

  // Converte gli argomenti da stringa a numero
  const unsigned int refinement_level = std::stoul(argv[1]);
  const double       deltat           = std::stod(argv[2]);

  const unsigned int degree = 1;
  const double       T      = 2.0;
  const double       theta  = 0.5;

  // Passa i nuovi argomenti al costruttore di Heat
  Heat problem(degree, refinement_level, T, deltat, theta);

  problem.setup();
  problem.solve();

  return 0;
}