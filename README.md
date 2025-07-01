# Space and Time adaptivity

#### Numerical Methods for Partial Differential Equations – Project 2024/25 - Politecnico di Milano

This project explores **adaptive methods** in both **space and time** for solving partial differential equations. We focus on the heat equation with a localized, time-dependent forcing term characterized by sharp, frequent impulses. The goal is to capture steep solution features accurately while minimizing computational cost through dynamic mesh and time step refinement based on error estimates. We implemented an adaptive solver using the ```deal.II``` library and evaluated its effectiveness in terms of accuracy and efficiency.


# TODO add cool image
![cool_image]()


**Students**
* [Leonardo Arnaboldi](https://github.com/leo-arnaboldi)
* [Luca Donato](https://github.com/lucacris72)
* [Tommaso Crippa](https://github.com/crippius)


**Professor**: *Alfio Maria Quarteroni*


**Assistant Professor**: *Michele Bucelli*

## Strong formulation
$$
\left\{
\begin{aligned}
\frac{\partial u}{\partial t} - \nabla \cdot (\mu \nabla u) &= f && \text{in } \Omega \times (0,T), \\
\mu \nabla u \cdot \mathbf{n} &= 0 && \text{on } \partial \Omega \times (0,T), \\
u &= 0 && \text{in } \Omega \times \{0\},
\end{aligned}
\right.
$$

$$
f(\mathbf{x},t) = g(t)h(\mathbf{x}), \quad
g(t) = \frac{\exp(-a \cos(2N \pi t))}{\exp(a)}, \quad
h(\mathbf{x}) = \exp\left(-\left(\frac{|\mathbf{x} - \mathbf{x}_0|}{\sigma}\right)^2\right),
$$

$$
\text{where } a > 0,\; N \in \mathbb{N},\; x_0 \in \Omega,\; \text{and } \sigma > 0.
$$


## Prerequisites

* **deal.II** ≥ 9.0.
* **C++** compiler (C++11 or later).
* **CMake** ≥ 3.12.
* **Python** ≥3.6



## Compiling
To build the executable, make sure you have loaded the needed modules with
```bash
$ module load gcc-glibc dealii
```
Then run the following commands:
```bash
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Execution

# TODO make program take parameters from terminal

The program can be executed from the `build` folder through

```bash
$ ./main ... [optional]
```
where
- ...
- ...
- Optional parameters:
    - ...
    - ...
    - ...

## Project Structure

# TODO check if correct

```text
root/                                      
├── src/
│   ├── Heat.hpp   
│   ├── Heat.cpp    
│   └── main.cpp               
├── results/
│   ├── plot_results.ipynb        
│   └── results.csv
├── CMakeLists.txt 
└── report.pdf          
```