#include "Heat.hpp"

// Main function.
int
main(int argc, char *argv[])
{
  if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <parameter_file>" << std::endl;
      return 1;
    }

  try
  {
    ParameterHandler prm;

    Heat::declare_parameters(prm);
    prm.parse_input(argv[1]);
    Heat problem(prm);

    problem.setup();
    problem.solve();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}