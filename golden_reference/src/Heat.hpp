#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>


#include <deal.II/distributed/tria.h>

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
      return 1.0;
    }
  };

  // Function for the forcing term:
  //   f(x,t) = (∑ₖ Aₖ sin(2π νₖ t + φₖ))  *  (∑ᵢ exp(-||x-xᵢ||²/σ_spatial²))
  // Function for the forcing term (9 sources, relay activation in groups).
  class ForcingTerm : public Function<dim>
  {
  public:
    // Constructor
    explicit ForcingTerm(const double T_final)
      : Function<dim>(),
        T(T_final),
        // Heat sources parameters
        A{1.0, 0.5, 0.25},
        nu{1.0, 3.0, 5.0},
        phi{0.0, M_PI / 2.0, M_PI}
    {

      centers = {// Group 1: Short peaks
                 Point<dim>(0.1, 0.1, 0.1),
                 Point<dim>(0.9, 0.1, 0.9),
                 Point<dim>(0.5, 0.9, 0.5),
                 // Groupo 2: Medium peaks
                 Point<dim>(0.2, 0.8, 0.2),
                 Point<dim>(0.8, 0.2, 0.8),
                 Point<dim>(0.2, 0.2, 0.8),
                 // Group 3: Wide peaks
                 Point<dim>(0.5, 0.5, 0.5),
                 Point<dim>(0.1, 0.9, 0.1),
                 Point<dim>(0.9, 0.9, 0.1)};

      sigmas = {// Group 1: Short peaks
                0.015,
                0.020,
                0.018,
                // Group 2: Medium peaks
                0.08,
                0.07,
                0.09,
                // Group 3: Wide peaks
                0.20,
                0.22,
                0.18};
    }

    // Forcing term: f(x,t) = (∑ₖ Aₖ sin(2π νₖ t + φₖ)) * (∑ᵢ exp(-||x-xᵢ||²/σ_spatial²))
    virtual double value(const Point<dim> &p,
                         const unsigned int /*component*/ = 0) const override
    {
      const double t = this->get_time();

      
      double temporal_wave = 0.0;
      for (unsigned int k = 0; k < A.size(); ++k)
        temporal_wave += A[k] * std::sin(2.0 * M_PI * nu[k] * t + phi[k]) + A[k];

      double spatial_term = 0.0;

      // Activate sources in groups based on time t
      // 3 sources per group, total 9 sources
      if (t < T / 3.0)
        {
          for (unsigned int i = 0; i < 3; ++i)
            spatial_term += std::exp(-p.distance_square(centers[i]) /
                                     (sigmas[i] * sigmas[i]));
        }
      else if (t < 2.0 * T / 3.0)
        {
          for (unsigned int i = 3; i < 6; ++i)
            spatial_term += std::exp(-p.distance_square(centers[i]) /
                                     (sigmas[i] * sigmas[i]));
        }
      else
        {
          for (unsigned int i = 6; i < 9; ++i)
            spatial_term += std::exp(-p.distance_square(centers[i]) /
                                     (sigmas[i] * sigmas[i]));
        }

      return temporal_wave * spatial_term;
    }

  public: 
    const double            T; 
    std::vector<double>     A, nu, phi;
    std::vector<Point<dim>> centers;
    std::vector<double>     sigmas;
  };

  // Function for the initial condition.
  class FunctionU0 : public Function<dim>
  {
  public:
    virtual double
    value(const Point<dim> &/*p*/,
          const unsigned int /*component*/ = 0) const override
    {
      return 0.0;
    }
  };

  // Constructor. We provide the final time, time step Delta t and theta method
  // parameter as constructor arguments.
  Heat(const unsigned int &r_,
       const unsigned int &refinement_level_,
       const double       &T_,
       const double       &deltat_,
       const double       &theta_)
    : mpi_size(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , mpi_rank(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank == 0)
    , forcing_term(T_)
    , T(T_)
    , r(r_)
    , refinement_level(refinement_level_)
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

  const unsigned int refinement_level;

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