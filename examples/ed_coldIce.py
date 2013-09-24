import numpy as np
import matplotlib.pyplot as plt
import femtoybox.dirichlet as ft
from scipy.sparse.linalg import spsolve


# this version modifies coldIce.py
# substantive changes:
#    * switch sign of "F" in function solveHomo()
#    * "particular_solution()" makes no sense to me as a name; it is
#      not a *solution* but merely a boundary-value-satisfying function
#    * using the b-v-satisfying function requires changing the problem,
#      so I have added "Sprime" into solveNonHomo() to compensate S when
#      using bvSatisfied()
# formal changes:
#    * remove V and S from parameter class
#    * pass class P containing parameters, not individual parameters
#    * show three cases in one figure:  S=0,V=0;  S>0,V=0;  S=0,V<0
#    * print errors instead of in figures
# WARNINGS:
#    * this version assumes V is constant at several stages
#    * the exact solution exactSolutionNoadvect() assumes S is constant;
#      if it is not constant we must either fix or abandon
#      exactSolutionNoadvect()


class Params():
  def __init__(self):
    """typical values taken from Aschwanden et. al. 2012"""

    # properties of ice
    self.rho = 910.0   # kg m-3
    self.c = 2009.0    # J K-1 kg-1
    self.k = 2.1       # J m-1 K-1 s-1

    # geometry of particular problem
    self.ell = 1000.0  # m

    # boundary values of particular problem
    self.T_0 = -10.0   # C


def exactSolutionNoadvect(P,S_0):
  """Returns a function which is the exact solution of the nonhomogeneous
  boundary value problem with zero advection and constant source:
    0 = (k u_z)_z + S_0
    u(ell) = T_0
    u(0) = 0
  """
  def u(z):
    return (S_0 / (2.0*P.k)) * (P.ell - z) * z + (P.T_0 / P.ell) * z
  return u


def critStrainNoAdvect(P):
  """Assuming V0 = 0, returns the critical S0, so that u(z) <= 0
  for any 0 <= z <= ell. See notes for derivation.
  """
  critS0 = -2.0 * P.T_0 * P.k / (P.ell)**2
  return critS0


def exactSolutionNosource(P,V_0):
  """Returns a function which is the exact solution of the nonhomogeneous
  boundary value problem with zero source and constant advection:
    rho c (V_0 u)_z = (k u_z)_z
    u(ell) = T_0
    u(0) = 0
  """
  if V_0 == 0.0:    # special case; avoids division by zero
    def u(z):
      return (P.T_0 / P.ell) * z
  else:
    gamma = P.rho * P.c * V_0 / P.k
    def u(z):
      return P.T_0 * (np.exp(gamma * z) - 1.0) / (np.exp(gamma * P.ell) - 1.0)
  return u


def bvSatisfied(P):
  """Returns a linear function which satisfies the boundary values
    u(ell) = T_0
    u(0) = 0
  """
  def u(z):
    return (P.T_0 / P.ell) * z
  return u


def solveHomo(P, S, V, grid):
  """Solve by FEM the homogeneous boundary value problem
    (*)  rho c (V u)_z = (k u_z)_z + S
         u(ell) = 0
         u(0) = 0
  WARNING: This version assumes V is constant.
  """
  A = - P.k * ft.stiffness(grid)
  B = P.rho * P.c * V(0.0) * ft.advection(grid)
  C = A + B
  F = - ft.intF(S, grid)
  n = len(grid) - 2
  y = np.zeros((n+2,))
  y.transpose()
  y[1:n+1] = spsolve(C, F)
  return y


def solveNonHomo(P, S, V, grid):
  """Solve by FEM the differential equation (*) but with boundary values
    u(ell) = T_0
    u(0) = 0
  WARNING: This version assumes V is constant!
  """
  Sprime = lambda z: S(z) - P.rho * P.c * V(0.0) * P.T_0 / P.ell
  y = solveHomo(P, Sprime, V, grid)
  upart = bvSatisfied(P)
  y = y + upart(grid)
  return y


def main():
  P = Params()

  fig = plt.figure()
  grid = np.linspace(0,P.ell,30)
  gridFine = np.linspace(0,P.ell,300)

  JJ = 3   # do and show 3 cases
  for j in range(JJ):
    if j == 0:
      S0 = 0.0
      V0 = 0.0
    elif j == 1:
      S0 = critStrainNoAdvect(P)       # strain heating in FIXME units
      V0 = 0.0
    elif j == 2:
      S0 = 0.0
      spera = 31556926.0 # seconds in a year
      V0 = - 0.2 / spera # 20 cm per year downward
      # NOTE:  V0 = -0.5/spera  causes an EXPECTED instability, I believe;
      #        do you agree?
    else:
      print 'ERROR: unexpected case'

    if j <= 1:
      uexact = exactSolutionNoadvect(P,S0)
    elif j == 2:
      uexact = exactSolutionNosource(P,V0)

    # for now: S and V constant in all cases
    S = lambda x: S0
    V = lambda x: V0
    U = solveNonHomo(P, S, V, grid)

    ax = fig.add_subplot(JJ,1,j+1)
    ax.plot(uexact(gridFine), gridFine, label="Exact")
    ax.plot(U, grid, 'o', label="FEM")
    ax.set_xlabel("u")
    ax.set_ylabel("z")
    ax.legend()

    print 'error when S = %.3e and V = %.3e is   %.3e'\
          % (S0, V0, max(abs(U - uexact(grid))))

  plt.show()
  fig.savefig('ed_coldIce_results_new.png')


if __name__=="__main__":
  main()
