#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree         = 1;
  const double T      = 1.0;
  const double deltat = 0.05;
  const double theta  = 1.0;
  const unsigned int n_refinements = 3;

  Heat problem(degree, T, deltat, theta, n_refinements);

  problem.setup();
  problem.solve();

  return 0;
}