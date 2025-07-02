#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  const unsigned int degree         = 1;

  const double T      = 2.0;
  const double deltat = 0.01;
  const double theta  = 0.5;

  Heat problem(degree, T, deltat, theta);

  problem.setup();
  problem.solve();

  return 0;
}