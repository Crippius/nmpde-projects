#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  const unsigned int degree         = 1;
  const double T      = 2.0;         // Longer simulation time
  const double deltat = 0.05;
  const double theta  = 0.5;         // Change to Crank-Nicolson scheme
  const unsigned int n_refinements = 2; // Start with less global refinements to make adaptivity more visible

  Heat problem(degree, T, deltat, theta, n_refinements);

  problem.setup();
  problem.solve();

  return 0;
}