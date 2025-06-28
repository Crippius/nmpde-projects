#ifndef HEAT_HPP
#define HEAT_HPP

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>

#include <vector>
#include <cmath>
#include <chrono>

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

  // TODO: make these selections on main
  // Function for the forcing term f(x,t) =
  //   (∑k A_k sin(2π ν_k t + φ_k))  *  (∑i exp(-||x-x_i||²/σ²))
  class ForcingTerm : public Function<dim> {
  public:
    ForcingTerm()
      : Function<dim>(),
        A{1.0, 0.5, 0.25},
        nu{1.0, 3.0, 5.0},
        phi{1.0, 0.0, 1.0},
        sigma_spatial(0.1)
    {
      // Centri di M = 9 sorgenti su griglia 3×3, in z = 0.5 per il piano mid‐slice
      centers = {
        Point<dim>(0.25,0.25,0.5), Point<dim>(0.25,0.50,0.5), Point<dim>(0.25,0.75,0.5),
        Point<dim>(0.50,0.25,0.5), Point<dim>(0.50,0.50,0.5), Point<dim>(0.50,0.75,0.5),
        Point<dim>(0.75,0.25,0.5), Point<dim>(0.75,0.50,0.5), Point<dim>(0.75,0.75,0.5)
      };
    }

    virtual double value(const Point<dim> &p,
                        const unsigned int /*component*/=0) const override
    {
      const double t = this->get_time();
      // parte temporale
      double temporal = 0.0;
      for (unsigned int k = 0; k < A.size(); ++k)
        temporal += A[k] * std::sin(2.0*M_PI*nu[k]*t + phi[k]);
      // parte spaziale
      double spatial = 0.0;
      for (const auto &c : centers)
        spatial += std::exp(- (p.distance(c)*p.distance(c))
                            / (sigma_spatial*sigma_spatial));
      return temporal * spatial;
    }

  private:
    std::vector<double>      A, nu, phi;
    std::vector<Point<dim>>  centers;
    double                   sigma_spatial;
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

  Heat(const unsigned int &r_,
       const double       &T_,
       const double       &deltat_,
       const double       &theta_)
    : T(T_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
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

  // Refine the grid based on solution error estimation.
  void
  refine_grid();

  // Setup for constraints (needed for hanging nodes in adaptive refinement)
  void setup_constraints();

  // Estimate the time discretization error for the current step
  double estimate_time_error(const double &time, const Vector<double> &prev_solution, double trial_deltat);

  // Update the time step based on the error estimation
  void update_deltat(double time, Vector<double> &prev_solution);

  // Output.
  void
  output(const unsigned int &time_step) const;
  


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

  // Time step. (not const -> time adaptativity)
  double deltat;

  // Theta parameter of the theta method.
  const double theta;

  // Mesh.
  Triangulation<dim> mesh;

  // Finite element space.
  std::unique_ptr<FiniteElement<dim>> fe;

  // Quadrature formula.
  std::unique_ptr<Quadrature<dim>> quadrature;

  // DoF handler.
  DoFHandler<dim> dof_handler;
  
  // Constraints for hanging nodes in adaptive refinement
  AffineConstraints<double> constraints;

  // Mass matrix M / deltat.
  SparseMatrix<double> mass_matrix;

  // Stiffness matrix A.
  SparseMatrix<double> stiffness_matrix;

  // Matrix on the left-hand side (M / deltat + theta A).
  SparseMatrix<double> lhs_matrix;

  // Matrix on the right-hand side (M / deltat - (1 - theta) A).
  SparseMatrix<double> rhs_matrix;

  // Right-hand side vector in the linear system.
  Vector<double> system_rhs;

  // System solution
  Vector<double> solution;
  
  // Sparsity pattern
  SparsityPattern sparsity_pattern;

  // Space Adaptativity Parameters. ///////////////////////////////////////////////////////////

  // Initial mesh refinement level
  const unsigned int n_global_refinements = 2;
  // # Steps for each space adaptation
  const unsigned int refinement_interval = 5;
  // Top X% for refinement
  const double refinement_percent = 0.1;
  // Bottom X% for coarsening
  const double coarsening_percent = 0.9;

  // Time Adaptativity Parameters. ///////////////////////////////////////////////////////////

  // Lower bound for time error
  double time_error_lower_bound = 0.0005; 
  // Upper bound for time error
  double time_error_upper_bound = 0.002;   
  // Minimum allowed time step
  double min_deltat = 1e-4;       
  // Maximum allowed time step
  double max_deltat = 0.2;              
  // # Steps for each time adaptation
  unsigned int time_adapt_interval = 1; 

  // PERFORMANCE TRACKING
  std::chrono::duration<double> time_total{0.0};

  std::chrono::duration<double> time_refine{0.0};

  std::chrono::duration<double> time_assemble_matrices{0.0};

  std::chrono::duration<double> time_assemble_rhs{0.0};

  std::chrono::duration<double> time_solve_step{0.0};

  unsigned int n_time_steps{0};

  unsigned int n_refinements{0};

  unsigned long sum_dofs{0};         // accumulate DoFs over time
  unsigned int  num_assemblies{0};   // how many assemble_matrices() calls

  // weights (seconds per unit)
  double beta {1e-6};   // per DoF
  double gamma{1e-2};   // per time step
  double delta{0.1};    // per assembly
  double zeta {0.5};    // per refinement
};

#endif