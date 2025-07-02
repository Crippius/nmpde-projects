#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// Class representing the non-linear diffusion problem.
class Heat
{
public:
  // Physical dimension (1D, 2D, 3D)
  static constexpr unsigned int dim = 3;

  // Function for the mu coefficient.
  class FunctionMu : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> & /*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.1;
    }
  };

  // Function for the forcing term:
  //   f(x,t) = (∑ₖ Aₖ sin(2π νₖ t + φₖ))  *  (∑ᵢ exp(-||x-xᵢ||²/σ_spatial²))
  class ForcingTerm : public Function<dim>
  {
  public:
    ForcingTerm()
      : Function<dim>(),
        A{1.0, 0.5, 0.25},
        nu{1.0, 3.0, 5.0},
        phi{0.0, 0.0, 0.0},
        sigma_spatial(0.1)
    {
      centers = {
        Point<dim>(0.25,0.25,0.5), Point<dim>(0.25,0.50,0.5), Point<dim>(0.25,0.75,0.5),
        Point<dim>(0.50,0.25,0.5), Point<dim>(0.50,0.50,0.5), Point<dim>(0.50,0.75,0.5),
        Point<dim>(0.75,0.25,0.5), Point<dim>(0.75,0.50,0.5), Point<dim>(0.75,0.75,0.5)
      };
    }

    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      const double t = this->get_time();
      // temporal part
      double temporal = 0.0;
      for (unsigned int k = 0; k < A.size(); ++k)
        temporal += A[k] * std::sin(2.0*M_PI*nu[k]*t + phi[k]) + A[k];
      // spatial part
      double spatial = 0.0;
      for (const auto &c : centers)
        spatial += std::exp(- p.distance(c)*p.distance(c)
                            / (sigma_spatial*sigma_spatial));
      return temporal * spatial;
    }

  private:
    std::vector<double>     A, nu, phi;
    std::vector<Point<dim>> centers;
    double                  sigma_spatial;
  };

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &p,
          const unsigned int /*component*/ = 0) const override
    {
      return p[0] * (1.0 - p[0]) * p[1] * (1.0 - p[1]) * p[2] * (1.0 - p[2]);
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Heat(const unsigned int &r_,
       const double       &T_,
       const double       &deltat_,
       const double       &theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , T(T_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , mesh(MPI_COMM_WORLD)
  {}

  // Initialization.
  void
  setup();

  // Solve the problem.
  void
  solve();

protected:
  // Create the cube mesh
  void
  create_mesh();

  // Assemble the mass and stiffness matrices.
  void
  assemble_matrices();

  // Assemble the right-hand side of the problem.
  void
  assemble_rhs(const double &time);

  // Solve the problem for one time step.
  void
  solve_time_step();

  // Output.
  void
  output(const unsigned int &time_step) const;

  // MPI parallel. /////////////////////////////////////////////////////////////

  // Number of MPI processes.
  const unsigned int mpi_size;

  // This MPI process.
  const unsigned int mpi_rank;

  // Parallel output stream.
  ConditionalOStream pcout;

  // Problem definition. ///////////////////////////////////////////////////////

  // mu coefficient.
  FunctionMu mu;

  // Forcing term.
  ForcingTerm forcing_term;

  // Initial condition.
  FunctionU0 u_0;

  // Final time.
  const double T;

  // Discretization. ///////////////////////////////////////////////////////////

  // Polynomial degree.
  const unsigned int r;

  // Time step.
  const double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Mesh.
  parallel::distributed::Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;

  // DoFs owned by current process.
  IndexSet locally_owned_dofs;

  // DoFs relevant to the current process (including ghost DoFs).
  IndexSet locally_relevant_dofs;

  // Mass matrix M / deltat.
  TrilinosWrappers::SparseMatrix mass_matrix;

  // Stiffness matrix A.
  TrilinosWrappers::SparseMatrix stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  TrilinosWrappers::SparseMatrix lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  TrilinosWrappers::SparseMatrix rhs_matrix;

  // Right-hand side vector in the linear system.
  TrilinosWrappers::MPI::Vector system_rhs;

  // System solution (without ghost elements).
  TrilinosWrappers::MPI::Vector solution_owned;

  // System solution (including ghost elements).
  TrilinosWrappers::MPI::Vector solution;

  // PERFORMANCE METRICS
  std::chrono::duration<double> wall_clock_time{0.0}; // total run time

  // resource counters
  unsigned long sum_dofs{0};     // cumulative DoFs over all steps
  unsigned int  num_time_steps{0};
  unsigned int  num_assemblies{0};

  // weights: seconds per unit
  double beta {1e-6};  // per DoF
  double gamma{1e-2};  // per time step
  double delta{0.1};   // per assembly
};

#endif