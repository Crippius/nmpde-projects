#include "Heat.hpp"
#include <chrono>


void
Heat::declare_parameters(ParameterHandler &prm)
{
    prm.enter_subsection("Adaptivity Control");
  {
    prm.declare_entry("Enable space adaptivity", "true", Patterns::Bool(), 
                      "If true, enables adaptive mesh refinement.");
    prm.declare_entry("Enable time adaptivity", "true", Patterns::Bool(), 
                      "If true, enables adaptive time stepping.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Discretization");
  {
    prm.declare_entry("Degree", "1", Patterns::Integer(0), "Polynomial degree of FE");
    prm.declare_entry("Global refinements", "2", Patterns::Integer(0), 
                      "Number of global refinements for uniform meshes.");
    prm.declare_entry("Final time", "2.0", Patterns::Double(0.0), "Final time T");
    prm.declare_entry("Initial deltat", "0.05", Patterns::Double(0.0), "Initial time step size");
    prm.declare_entry("Theta", "0.5", Patterns::Double(0.0, 1.0), "Theta for the time-stepping scheme");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space Adaptivity");
  {
    prm.declare_entry("Refinement interval", "5", Patterns::Integer(1), "Steps between mesh refinements");
    prm.declare_entry("Refinement fraction", "0.1", Patterns::Double(0.0, 1.0), "Top fraction of cells to refine");
    prm.declare_entry("Coarsening fraction", "0.9", Patterns::Double(0.0, 1.0), "Bottom fraction of cells to coarsen");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time Adaptivity");
  {
    prm.declare_entry("Adaptivity interval", "1", Patterns::Integer(1), "Steps between time error checks");
    prm.declare_entry("Error lower bound", "0.0005", Patterns::Double(0.0), "Lower bound for time error");
    prm.declare_entry("Error upper bound", "0.002", Patterns::Double(0.0), "Upper bound for time error");
    prm.declare_entry("Min deltat", "1e-4", Patterns::Double(0.0), "Minimum allowed time step");
    prm.declare_entry("Max deltat", "0.2", Patterns::Double(0.0), "Maximum allowed time step");
  }
  prm.leave_subsection();
}

void
Heat::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Adaptivity Control");
  {
    enable_space_adaptivity = prm.get_bool("Enable space adaptivity");
    enable_time_adaptivity  = prm.get_bool("Enable time adaptivity");
  }
  prm.leave_subsection();

  prm.enter_subsection("Discretization");
  {
    r      = prm.get_integer("Degree");
    n_global_refinements = prm.get_integer("Global refinements");
    T      = prm.get_double("Final time");
    deltat = prm.get_double("Initial deltat");
    theta  = prm.get_double("Theta");
  }
  prm.leave_subsection();

  prm.enter_subsection("Space Adaptivity");
  {
    refinement_interval = prm.get_integer("Refinement interval");
    refinement_percent  = prm.get_double("Refinement fraction");
    coarsening_percent  = prm.get_double("Coarsening fraction");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time Adaptivity");
  {
    time_adapt_interval    = prm.get_integer("Adaptivity interval");
    time_error_lower_bound = prm.get_double("Error lower bound");
    time_error_upper_bound = prm.get_double("Error upper bound");
    min_deltat             = prm.get_double("Min deltat");
    max_deltat             = prm.get_double("Max deltat");
  }
  prm.leave_subsection();
}


Heat::Heat(ParameterHandler &prm)
  : forcing_term(0.0) // Inizializzazione temporanea, T verrà sovrascritto
{
  // Leggi i parametri dal file e popola le variabili membro
  parse_parameters(prm);
  
  // Ora che T è noto, aggiorna il membro T della classe ForcingTerm
  const_cast<double&>(forcing_term.T) = T;
}


void
Heat::create_mesh()
{
  std::cout << "Creating cube mesh" << std::endl;
  
  GridGenerator::hyper_cube(mesh, 0.0, 1.0);
  mesh.refine_global(n_global_refinements);
  
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
      constraints.distribute_local_to_global(cell_rhs, dof_indices, system_rhs);
    }

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
Heat::refine_grid()
{

  std::cout << std::endl <<  "[Space adaptivity] Performing adaptive mesh refinement" << std::endl;

  std::cout << " Refining grid based on error estimation" << std::endl;
  std::cout << "  Number of active cells before refinement: " 
            << mesh.n_active_cells() << std::endl;
  
  Vector<float> estimated_error_per_cell(mesh.n_active_cells());

  KellyErrorEstimator<dim>::estimate(dof_handler,
                                    QGauss<dim - 1>(r + 1),
                                    {}, 
                                    solution,
                                    estimated_error_per_cell);
  
  GridRefinement::refine_and_coarsen_fixed_number(mesh,
                                                 estimated_error_per_cell,
                                                 refinement_percent,
                                                 coarsening_percent);
  
  SolutionTransfer<dim> solution_transfer(dof_handler);
  mesh.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(solution);
  mesh.execute_coarsening_and_refinement();
  
  std::cout << "  Number of active cells after refinement: " 
            << mesh.n_active_cells() << std::endl;
  
  dof_handler.distribute_dofs(*fe);
  std::cout << "  Number of DoFs after refinement = " << dof_handler.n_dofs() << std::endl;
  
  setup_constraints();
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);
  
  mass_matrix.reinit(sparsity_pattern);
  stiffness_matrix.reinit(sparsity_pattern);
  lhs_matrix.reinit(sparsity_pattern);
  rhs_matrix.reinit(sparsity_pattern);
  
  Vector<double> new_solution(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  
  solution_transfer.interpolate(solution, new_solution);
  solution.reinit(dof_handler.n_dofs());
  solution = new_solution;
  constraints.distribute(solution);

  std::cout << "  Reassembling matrices for the refined mesh" << std::endl;
  assemble_matrices();
}

void
Heat::setup_constraints()
{
  std::cout << "Setting up constraints" << std::endl;
  
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.close();
}

double Heat::estimate_time_error(const double &time, const Vector<double> &prev_solution, double trial_deltat)
{
  Vector<double> backup_solution = solution;
  double backup_deltat = deltat;

  deltat = trial_deltat;
  solution = prev_solution;
  assemble_rhs(time + trial_deltat);
  solve_time_step();
  Vector<double> sol_big_step = solution;

  deltat = trial_deltat / 2.0;
  solution = prev_solution;
  assemble_rhs(time + deltat);
  solve_time_step();
  Vector<double> sol_half_step = solution;
  assemble_rhs(time + 2.0 * deltat);
  solve_time_step();
  Vector<double> sol_two_half_steps = solution;

  sol_big_step -= sol_two_half_steps;
  double error = sol_big_step.l2_norm();

  solution = backup_solution;
  deltat = backup_deltat;

  return error;
}

void Heat::update_deltat(double time, Vector<double> &prev_solution) {
  bool step_accepted = false;
  double trial_deltat = deltat;
  double local_time = time;
  double prev_deltat = 0;

  prev_solution = solution;

  while (!step_accepted)
  {
    double time_error = estimate_time_error(local_time, prev_solution, trial_deltat);
    if (time_error < time_error_lower_bound && trial_deltat * 2.0 <= max_deltat && trial_deltat * 2.0 != prev_deltat)
    {
      prev_deltat = trial_deltat;
      trial_deltat *= 2.0;
      std::cout << "[Time adaptivity] Time error " << time_error << " < lower bound. Increasing deltat to " << trial_deltat << std::endl;
    }
    else if (time_error > time_error_upper_bound && trial_deltat / 2.0 >= min_deltat && trial_deltat / 2.0 != prev_deltat)
    {
      prev_deltat = trial_deltat;
      trial_deltat /= 2.0;
      std::cout << "[Time adaptivity] Time error " << time_error << " > upper bound. Decreasing deltat to " << trial_deltat << std::endl;
    }
    else
    {
      deltat = trial_deltat;
      step_accepted = true;
      std::cout << "[Time adaptivity] Time error " << time_error << " -> reasonable. Not changing anything" << std::endl;
    }
  }
}

void
Heat::solve()
{
  // --- start wall-clock timer ---
  auto t_total_start = std::chrono::high_resolution_clock::now();

  // --- reset resource counters ---
  n_time_steps   = 0;
  unsigned int num_assemblies = 0;
  unsigned int n_refinements  = 0;

  auto t0 = std::chrono::high_resolution_clock::now();
  assemble_matrices();
  auto t1 = std::chrono::high_resolution_clock::now();
  time_assemble_matrices += t1 - t0;
  ++num_assemblies; 

  std::cout << "===============================================" << std::endl;

  {
    std::cout << "Applying the initial condition" << std::endl;
    VectorTools::interpolate(dof_handler, u_0, solution);
    output(0);
    std::cout << "-----------------------------------------------" << std::endl;
  }
  
  unsigned int time_step = 0;
  double time = 0;

  while (time < T)
  {
    ++n_time_steps;
    if (enable_space_adaptivity && time_step > 0 && time_step % refinement_interval == 0){
      auto tr0 = std::chrono::high_resolution_clock::now();
      refine_grid();
      auto tr1 = std::chrono::high_resolution_clock::now();
      time_refine += tr1 - tr0;
      n_refinements++;
      ++num_assemblies;
    }
      
    if (enable_time_adaptivity && time_step > 0 && time_step % time_adapt_interval == 0)
      update_deltat(time, solution);
    
    time += deltat;
    if (time > T) {
        deltat -= (time - T);
        time = T;
    }

    ++time_step;
    std::cout << "n = " << std::setw(3) << time_step << ", t = " << std::setw(5) << time << ", deltat = " << deltat << ":" << std::flush;

    t0 = std::chrono::high_resolution_clock::now();
    assemble_rhs(time);
    t1 = std::chrono::high_resolution_clock::now();
    time_assemble_rhs += t1 - t0;

    t0 = std::chrono::high_resolution_clock::now();
    solve_time_step();
    t1 = std::chrono::high_resolution_clock::now();
    time_solve_step += t1 - t0;

    output(time_step);
  }

  auto t_total_end = std::chrono::high_resolution_clock::now();
  time_total = t_total_end - t_total_start;

  // --- Calcola e stampa le nuove metriche ---
  compute_and_print_metrics();
}

void
Heat::compute_and_print_metrics() const
{
  // 1. RACCOLTA DEI DATI DI PERFORMANCE
  const double total_time = time_total.count();
  const double n_dofs = dof_handler.n_dofs();
  const double h_min = GridTools::minimal_cell_diameter(mesh);

  // 2. CALCOLO DELLE METRICHE
  const double r_res       = h_min / n_dofs;
  const double r_t_per_dof = total_time / n_dofs;

  // 3. STAMPA DEI RISULTATI
  std::cout << "\n===============================================" << std::endl;
  std::cout << "=== Performance Metrics Summary ===" << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "Total Wall-clock time (t):              " << total_time << " s" << std::endl;
  std::cout << "Final Degrees of Freedom (n_Omega):     " << n_dofs << std::endl;
  std::cout << "Minimum cell diameter (h):              " << h_min << std::endl;
  std::cout << "-----------------------------------------------" << std::endl;
  std::cout << "Resolution & Resource Metrics:" << std::endl;
  std::cout << "  - r_res (h / n_Omega):                " << r_res << std::endl;
  std::cout << "  - r_t-per-DOF (t / n_Omega):          " << r_t_per_dof << " s/DOF" << std::endl;
  std::cout << "===============================================" << std::endl;
}