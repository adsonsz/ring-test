#include <iostream>
#include <string>
#include <fstream>
#include <cmath>

// Grid values.
int N = 200;
int L = 8.0;
int R = 2*L;

// Source values for the current ring.
float current = 1.0;    // The total current of the ring.
float height = 0.0;     // The z-position of the ring.
float a = 1.0;          // The radius of the ring.

// Calculate grid spacing.
float h = 2*L / (static_cast<float>(N)-1);


// Peforms a jacobi relaxation iteration.
// Dirichlet boundary at r=0, with A=0.
// Neumann boundary everywhere else.
void jacobi_vector_relaxation_neumann(float** grid, float** source, float** aux) {
    float hsq = h*h;

    // Copy the grid into the auxiliary grid `aux`
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            aux[i][j] = grid[i][j];
        }
    }


    // Dirichlet boundary at r=0.
    // That is, impose A=0 when r=0.
    for (int i = 0; i < N; ++i) {
        grid[i][0] = 0.0;
    }

    for (int j = 1; j < N-1; ++j) {
        // Define r.
        float r = h * static_cast<float>(j);
        float fac = 1.0 / (4.0 + hsq / (r*r));

        // Boundary z=-L
        grid[0][j] = fac * (
            aux[1][j] + aux[0][j+1] + aux[1][j] + aux[0][j-1] + 
            hsq / (2.0 * r * h) * (aux[0][j+1] - aux[0][j-1])
        );

        // Relax at the bulk.
        for (int i = 1; i < N-1; ++i) {
            grid[i][j] = fac * (
                aux[i+1][j] + aux[i][j+1] + aux[i-1][j] + aux[i][j-1] +
                hsq / (2.0 * r * h) * (aux[i][j+1] - aux[i][j-1]) +
                hsq * source[i][j]
            );
       } 

        // Boundary z=L
        grid[N-1][j] = fac * (
            aux[N-2][j] + aux[N-1][j+1] + aux[N-2][j] + aux[N-1][j-1] +
            hsq / (2.0 * r * h) * (aux[N-1][j+1] - aux[N-1][j-1])
        );
    }

    // Relax at boundary r=R.
    // Relax at corner [N-1, 0].
    float r = h * static_cast<float>(N-1);
    float fac = 1.0 / (4.0 + hsq / (r*r));
    grid[0][N-1] = fac * (aux[1][N-1] + aux[0][N-2] + aux[1][N-1] + aux[0][N-2]);

    // Relax at boundary r=R.
    for (int i = 1; i < N-1; ++i) {
        grid[i][N-1] = fac * (aux[i+1][N-1] + aux[i][N-2] + aux[i-1][N-1] + aux[i][N-2]);
    }

    // Relax at corner [N-1, N-1]
    grid[N-1][N-1] = fac * (aux[N-2][N-1] + aux[N-1][N-2] + aux[N-2][N-1] + aux[N-1][N-2]);
}


// Peforms a Gauss-seidel relaxation iteration.
// Dirichlet boundary at r=0, with A=0.
// Neumann boundary everywhere else.
void seidel_vector_relaxation_neumann(float** grid, float** source) {
    float hsq = h*h;

    // Dirichlet boundary at r=0.
    // That is, impose A=0 when r=0.
    for (int i = 0; i < N; ++i) {
        grid[i][0] = 0.0;
    }

    for (int j = 1; j < N-1; ++j) {
        // Define r.
        float r = h * static_cast<float>(j);
        float fac = 1.0 / (4.0 + hsq / (r*r));

        // Boundary z=-L
        grid[0][j] = fac * (
            grid[1][j] + grid[0][j+1] + grid[1][j] + grid[0][j-1] + 
            hsq / (2.0 * r * h) * (grid[0][j+1] - grid[0][j-1])
        );

        // Relax at bulk.
        for (int i = 1; i < N-1; ++i) {
            grid[i][j] = fac * (
                grid[i+1][j] + grid[i][j+1] + grid[i-1][j] + grid[i][j-1] +
                hsq / (2.0 * r * h) * (grid[i][j+1] - grid[i][j-1]) +
                hsq * source[i][j]
            );
       } 

        // Boundary z=L
        grid[N-1][j] = fac * (
            grid[N-2][j] + grid[N-1][j+1] + grid[N-2][j] + grid[N-1][j-1] +
            hsq / (2.0 * r * h) * (grid[N-1][j+1] - grid[N-1][j-1])
        );
    }

    // Relax at boundary r=R.
    // Relax at corner [N-1, 0].
    float r = h * static_cast<float>(N-1);
    float fac = 1.0 / (4.0 + hsq / (r*r));
    grid[0][N-1] = fac * (grid[1][N-1] + grid[0][N-2] + grid[1][N-1] + grid[0][N-2]);

    // Relax at boundary r=R.
    for (int i = 1; i < N-1; ++i) {
        grid[i][N-1] = fac * (grid[i+1][N-1] + grid[i][N-2] + grid[i-1][N-1] + grid[i][N-2]);
    }

    // Relax at corner [N-1, N-1]
    grid[N-1][N-1] = fac * (grid[N-2][N-1] + grid[N-1][N-2] + grid[N-2][N-1] + grid[N-1][N-2]);
}


// Populates the source grid with values.
void current_ring_source(float** source) {
    float PI = 3.1415926535;

    // Identify the central portion of the current ring.
    /*int nz = std::round((height + L) / h);
    int nr = std::round(a / h);
    float dv = 2.0 * PI * a * h*h;
    source[nz][nr] = current / dv;*/

    // Gaussian profile.
    // J(r,z) = amplitude * exp[-(r-r0)**2 - z^2   /   2 sigma]
    float amplitude = 1.0;
    float sigma = 3*h;
    float mean_r = a;
    float mean_z = height;

    float total_current = 0.0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            // Calculate (r,z) coordinates from (i, j).
            float r = h * static_cast<float>(j);
            float z = -L + h * static_cast<float>(i);

            // Calculate gaussian current density, J.
            float arg = (r-mean_r) * (r-mean_r) + (z-mean_z) * (z-mean_z);
            float J = amplitude * std::exp(-arg / (2.0 * sigma * sigma));

            // Set the source.
            source[i][j] = J;

            // Increment total current by numerical integration.
            // I = int J(r,z) dr dz. Therefore, dA = dr dz = h^2
            float area = h*h;
            total_current += J * area;
        }
    }


    // Make sure the current is the one selected.
    // Rescale the source, such that, the integral of current density is
    // the current chosen. This is equivalent of choosing an amplitude,
    // that makes the total integral to be the current.
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            source[i][j] *= current / total_current;
        }
    }
}


// Creates a file data.py, for storing the arrays and plotting.
void initfile() {
    std::ofstream file("data.py");
    file << "import numpy as np" << std::endl;
    file << std::endl;
    file.close();
}

// Saves the `grid` into the data.py, naming it as a variable called `var`.
void plot(float** grid, const std::string& var) {
    std::ofstream file("data.py", std::ios::app);
    file << var << " = np.array([";
    for (int i = 0; i < N-1; ++i) {
        for (int j = 0; j < N; ++j) {
            file << grid[i][j] << ", ";
        }
    }
    
    file << grid[N-1][0];
    for (int j = 1; j < N; ++j) file << ", " << grid[N-1][j];

    file << "]).reshape(" << N << ", " << N << ")";
    file << std::endl;
    file.close();
}

void calculate_current_ring(float** grid, float** source, float** aux) {
    // Creates data.py.
    initfile();

    // Populate the source grid.
    current_ring_source(source);

    // Calculate the answer by relaxation.
    for (int i = 0; i < N*N; ++i) {
        // seidel_vector_relaxation_neumann(grid, source);
        jacobi_vector_relaxation_neumann(grid, source, aux);
    }

    // Creates a variable called `grid` and stores it in data.py.
    // The grid contains the answer.
    plot(grid, "grid");
}


int main() {
    // Define the dimensions of the aux
    int rows = N;
    int cols = N;

    // Allocate grids.
    float **source = new float *[rows];
    float **grid = new float *[rows];
    float **aux = new float *[rows];
    for (int i = 0; i < rows; ++i) grid[i] = new float[cols];
    for (int i = 0; i < rows; ++i) aux[i] = new float[cols];
    for (int i = 0; i < rows; ++i) source[i] = new float[cols];


    // Calculate solution.
    calculate_current_ring(grid, source, aux);


    // Delete grids.
    for (int i = 0; i < rows; ++i) delete[] source[i];
    for (int i = 0; i < rows; ++i) delete[] grid[i];
    for (int i = 0; i < rows; ++i) delete[] aux[i];
    delete[] source;
    delete[] grid;
    delete[] aux;

    return 0;
}
