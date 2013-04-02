import femtoybox.dirichlet as femtoy
import numpy as np
import numpy.random
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def solveDirch(f, grid):
  """
  Returns an FEM solution to the problem
    -u'' u = f
  with Dirichlet BC's
  """
  A = femtoy.stiffness(grid)
  C = femtoy.zeroOrder(grid)
  F = femtoy.intF(f, grid)
  n = len(grid) - 2
  y = np.zeros((n+2, ))
  y.transpose()
  D = A+C
  y[1:n+1] = spsolve(D, F)
  return y

def main():
  J = 10
  grid = [np.linspace(0, 1, J+1)
      , (np.sin(np.linspace(-np.pi/2.,np.pi/2.,J+1))+1)/2.
      , np.zeros((J+1,))]
  a = numpy.random.RandomState(4)
  XX = a.rand(J-1)
  XX.sort()
  grid[2][1:J] = XX
  grid[2][J] = 1.

  fineGrid = np.linspace(0, 1, 200)

  f = lambda x: -12*(x**2) + x**4 - x
  u = lambda x: x**4 - x
  uExactFine = u(fineGrid)
  
  fig = plt.figure()

  for kk in range(3):
    U = solveDirch(f, grid[kk])

    ax = fig.add_subplot(3, 1, kk+1)
    ax.plot(grid[kk], U, 'o', label='FEM')
    ax.plot(fineGrid, uExactFine, label='Exact')
    ax.set_xlabel('x')
    ax.legend(loc=2)
    uExact = u(grid[kk])
    ax.text(0.4,-0.15, 'error = %f' % max(abs(U - uExact)))

  fig.show()

if __name__ == '__main__':
  main()
