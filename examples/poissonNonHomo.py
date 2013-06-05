import femtoybox.dirichlet as ft
import numpy as np
import numpy.random
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


def solveHomo(f, grid):
  """
  Solves Poisson's equation in 1-D
    - u'' = f
  via the Galerkin method, with homogeneous BC;s
  """
  A = ft.stiffness(grid)
  F = - ft.intF(f, grid)
  n = len(grid) - 2
  y = np.zeros((n+2, ))
  y.transpose()
  y[1:n+1] = spsolve(A, F)
  return y

def partSolution(grid, bcs):
  """
  Solves Lapace's equation
    u'' = 0
  with boundary conditions
    u(grid[0]) = bcs[0]
    u(grid[-1]) = bcs[1]
  on the grid.
  Note that this is the *exact* solution.
  """
  uPart = lambda x: (bcs[1] - bcs[0])*x + bcs[0]
  return uPart(grid)

def solveNonHomo(f, grid, bcs):
  homoSolution = solveHomo(f, grid)
  solution = homoSolution + partSolution(grid, bcs)
  return solution

def main():
  J = 3
  grid = np.linspace(0,1,J+1)
  fineGrid = np.linspace(0,1,200)
  f = lambda x: 1 - 2*(x**2)

  bcs = np.array([-2, 0.5])
  g = bcs[0]
  h =  (bcs[1] - bcs[0] - 1/3.0)

  u = lambda x: g + h*x + 0.5*(x**2) - (x**4)/6
  uExactFine = u(fineGrid)
  uExact = u(grid)

  uFEM = solveNonHomo(f, grid, bcs)

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.plot(grid, uFEM, 'o', label="FEM")
  ax.plot(fineGrid, uExactFine, label="Exact")
  ax.set_xlabel("x")
  ax.legend(loc=2)
  ax.text(0.0, 0.0, 'error = %g' % max(abs(uFEM - uExact)))

  fig.show()

if __name__ == '__main__':
  main()
