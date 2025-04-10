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

  // TODO: make these selections on main
  static constexpr double a = 1.0;
  static constexpr double N = 1.0; 
  static constexpr double sigma = 0.1; 
  static const Point<dim> x0; // initialized in Heat.cpp

  // Function for g(t)
  class GFunction : public Function<dim> {
  public:
      virtual double 
      value(const Point<dim> & /*p*/, 
            const unsigned int /*component*/= 0) const override 
      {
          const double t = this->get_time();
          return std::exp(-a * std::cos(2 * N * M_PI * t)) / std::exp(a);
      }
  };

  // Function for h(x)
  class HFunction : public Function<dim> {
  public:
      virtual double 
      value(const Point<dim> &p, 
            const unsigned int /*component*/= 0) const override 
      {
        return std::exp(-((p - x0).norm_square()) / (sigma * sigma));
      }
  };

  // Function for the forcing term f(x, t) = g(t) * h(x)
  class ForcingTerm : public Function<dim> {
  public:
      ForcingTerm() : Function<dim>() {}
      
      virtual double 
      value(const Point<dim> &p, 
            const unsigned int /*component*/= 0) const override 
      {
        // Use the time that was set for this function to set the time for g
        g.set_time(this->get_time());
        return g.value(Point<dim>()) * h.value(p);
      }
      
  private:
      mutable GFunction g;
      HFunction h;
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
       const double       &T_,
       const double       &deltat_,
       const double       &theta_,
       const unsigned int &n_refinements_ = 3)
    : T(T_)
    , r(r_)
    , deltat(deltat_)
    , theta(theta_)
    , n_refinements(n_refinements_)
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

  // Output.
  void
  output(const unsigned int &time_step) const;
  
  // Setup for constraints (needed for hanging nodes in adaptive refinement)
  void setup_constraints();

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

  // Number of global refinements
  const unsigned int n_refinements;

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
};

#endif