#include "Heat.hpp"

// x0 definition
const Point<Heat::dim> Heat::x0 = Point<Heat::dim>(0.5, 0.5, 0.5);

void
Heat::create_mesh()
{
  std::cout << "Creating cube mesh" << std::endl;
  
  GridGenerator::hyper_cube(mesh, 0.0, 1.0);
  mesh.refine_global(n_refinements);
  
  std::cout << "  Number of elements = " << mesh.n_active_cells()
        << std::endl;
}

void
Heat::setup()
{
  // Create the mesh.
  {
    std::cout << "Initializing the mesh" << std::endl;
    create_mesh();
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the finite element space.
  {
    std::cout << "Initializing the finite element space" << std::endl;

    fe = std::make_unique<FE_Q<dim>>(r);

    std::cout << "  Degree                     = " << fe->degree << std::endl;
    std::cout << "  DoFs per cell              = " << fe->dofs_per_cell
          << std::endl;

    quadrature = std::make_unique<QGauss<dim>>(r + 1);

    std::cout << "  Quadrature points per cell = " << quadrature->size()
          << std::endl;
  }

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the DoF handler.
  {
    std::cout << "Initializing the DoF handler" << std::endl;

    dof_handler.reinit(mesh);
    dof_handler.distribute_dofs(*fe);

    std::cout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;
  }

  // Set up constraints for hanging nodes
  setup_constraints();

  std::cout << "-----------------------------------------------" << std::endl;

  // Initialize the linear system.
  {
    std::cout << "Initializing the linear system" << std::endl;

    std::cout << "  Initializing the sparsity pattern" << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    std::cout << "  Initializing the matrices" << std::endl;
    mass_matrix.reinit(sparsity_pattern);
    stiffness_matrix.reinit(sparsity_pattern);
    lhs_matrix.reinit(sparsity_pattern);
    rhs_matrix.reinit(sparsity_pattern);

    std::cout << "  Initializing the system right-hand side" << std::endl;
    system_rhs.reinit(dof_handler.n_dofs());
    std::cout << "  Initializing the solution vector" << std::endl;
    solution.reinit(dof_handler.n_dofs());
  }
}

void
Heat::assemble_matrices()
{
  std::cout << "===============================================" << std::endl;
  std::cout << "Assembling the system matrices" << std::endl;

  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  mass_matrix      = 0.0;
  stiffness_matrix = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_mass_matrix      = 0.0;
      cell_stiffness_matrix = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // Evaluate coefficients on this quadrature node.
          const double mu_loc = mu.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                            fe_values.shape_value(j, q) /
                                            deltat * fe_values.JxW(q);

                  cell_stiffness_matrix(i, j) +=
                    mu_loc * fe_values.shape_grad(i, q) *
                    fe_values.shape_grad(j, q) * fe_values.JxW(q);
                }
            }
        }

      cell->get_dof_indices(dof_indices);

      // Apply constraints while assembling
      constraints.distribute_local_to_global(cell_mass_matrix, 
                                            dof_indices, 
                                            mass_matrix);
                                            
      constraints.distribute_local_to_global(cell_stiffness_matrix, 
                                           dof_indices, 
                                           stiffness_matrix);
    }

  // We build the matrix on the left-hand side of the algebraic problem (the one
  // that we'll invert at each timestep).
  lhs_matrix.copy_from(mass_matrix);
  lhs_matrix.add(theta, stiffness_matrix);

  // We build the matrix on the right-hand side (the one that multiplies the old
  // solution un).
  rhs_matrix.copy_from(mass_matrix);
  rhs_matrix.add(-(1.0 - theta), stiffness_matrix);
}

void
Heat::assemble_rhs(const double &time)
{
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q           = quadrature->size();

  FEValues<dim> fe_values(*fe,
                          *quadrature,
                          update_values | update_quadrature_points |
                            update_JxW_values);

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  system_rhs = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_rhs = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
        {
          // We need to compute the forcing term at the current time (tn+1) and
          // at the old time (tn). deal.II Functions can be computed at a
          // specific time by calling their set_time method.

          // Compute f(tn+1)
          forcing_term.set_time(time);
          const double f_new_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          // Compute f(tn)
          forcing_term.set_time(time - deltat);
          const double f_old_loc =
            forcing_term.value(fe_values.quadrature_point(q));

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              cell_rhs(i) += (theta * f_new_loc + (1.0 - theta) * f_old_loc) *
                             fe_values.shape_value(i, q) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(dof_indices);
      // Apply constraints while assembling
      constraints.distribute_local_to_global(cell_rhs, dof_indices, system_rhs);
    }

  // Add the term that comes from the old solution.
  rhs_matrix.vmult_add(system_rhs, solution);
}

void
Heat::solve_time_step()
{
  SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());

  SolverCG<Vector<double>> solver(solver_control);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(lhs_matrix, 1.0);

  solver.solve(lhs_matrix, solution, system_rhs, preconditioner);
  
  // Apply constraints
  constraints.distribute(solution);
  
  std::cout << "  " << solver_control.last_step() << " CG iterations" << std::endl;
}

void
Heat::output(const unsigned int &time_step) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "u");

  data_out.build_patches();

  std::ofstream output("output_" + std::to_string(time_step) + ".vtu");
  data_out.write_vtu(output);
}

void
Heat::solve()
{
  assemble_matrices();

  std::cout << "===============================================" << std::endl;

  // Apply the initial condition.
  {
    std::cout << "Applying the initial condition" << std::endl;

    VectorTools::interpolate(dof_handler, u_0, solution);

    // Output the initial solution.
    output(0);
    std::cout << "-----------------------------------------------" << std::endl;
  }

  unsigned int time_step = 0;
  double       time      = 0;

  while (time < T)
    {
      time += deltat;
      ++time_step;

      std::cout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5)
            << time << ":" << std::flush;

      assemble_rhs(time);
      solve_time_step();
      output(time_step);
    }
}

void
Heat::setup_constraints()
{
  std::cout << "Setting up constraints" << std::endl;
  
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();
}