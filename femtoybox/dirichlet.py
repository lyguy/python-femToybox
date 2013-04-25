import numpy as _np
import scipy.sparse as _sparse


def stiffnessEntry(grid, ii, jj):
  """Find \int \psi_i' \psi_j' dx where psi_i and \psi_j 
  are the respective entries of our pw-linear basis
  on the grid 'grid'

  TODO: Make this more friendly to negative indexing,
  right now it throws an error
  """

  ii, jj = max(ii, jj), min(ii, jj)
  L = len(grid)
  if ii >= L:
    raise IndexError('index %i ' %(ii)
      + 'is out of bounds for grid with size %i' % (L))
  
  if jj < 0:
    raise IndexError('index %i ' %(jj)
      + 'is out of bounds for grid with size %i' %(L))

  # Integrals on either side of grid point ii
  rightInt = lambda ii: - 1.0/(grid[ii + 1] - grid[ii])
  leftInt = lambda ii: - 1.0/(grid[ii] - grid[ii-1])

  def diag(ii):
    if ii == 0:
      return - rightInt(ii)
    elif ii == len(grid) - 1:
      return - leftInt(ii)
    else:
      return - leftInt(ii) - rightInt(ii)
  
  options = { 0: diag, -1: rightInt, 1: leftInt}
  
  try:
    return options[ii - jj](ii)
  except KeyError:
    return 0.0

def stiffness(grid):
  """Form the stiffness matrix A_{ij} = \int \phi_i' phi_j' 
  where phi_i is the i'th basis function for the Dirichlet problem 
  as described in the notes. 
  
  TODO: initialize A via coo, since the conversion
  to CRS is faster. It  will probably be easier to use some list
  comprehension to do this.

  """

  ell = _np.diff(grid) 
 # grid[1:len(grid)] - grid[0:len(grid) - 1]
  n = len(grid) - 2
  A = _sparse.dok_matrix((n, n))

  A[0, 0] = stiffnessEntry(grid, 1, 1)
  A[0, 1] = stiffnessEntry(grid, 1, 2)
  for ii in range(1, n - 1):
    A[ii, ii-1] = stiffnessEntry(grid, ii + 1, ii)
    A[ii, ii] = stiffnessEntry(grid, ii + 1, ii + 1)
    A[ii, ii+1] = stiffnessEntry(grid, ii + 1, ii + 2)
  A[n-1, n-2] = stiffnessEntry(grid, n, n-1)
  A[n-1, n-1] =  stiffnessEntry(grid, n, n)
  
  # Scipy's solvers require a CRS or CSC formated sparse matrix.
  # Since the solvers are more efficient for CRS, we'll use it.
  A = A.tocsr()
  return A

def zeroOrder(grid):
  """Form the mass matrix  C_{ij} = \int \phi_i+1 \phi_j+1 where 
  phi_i is the i'th basis function for the Dirichlet problem as described in the notes.

  TODO: * initialize C via coo, conversion to CRS is faster
        * do this integration via builtin quadrature methods
  """
  n = len(grid) - 2
  ell = _np.diff(grid)

  C = _sparse.dok_matrix((n,n))
  C[0, 0] = (ell[0] + ell[1]) / 3.
  C[0, 1] = ell[1] / 6.
  for ii in range(1, n-1):
    C[ii, ii - 1] = ell[ii] / 6.
    C[ii, ii] = (ell[ii] + ell[ii+1]) / 3.
    C[ii, ii + 1] = ell[ii+1] / 6.
  C[n-1,n-2] = ell[n-1]/6.
  C[n-1, n-1] = (ell[n-1] + ell[n])/3.

  # SciPy's solvers need a CRS or CRC sparse matrix
  C = C.tocsr()
  return C

def intF(f, grid):
  """
  Estimate \int f \phi_i using Gaussian quadrature

  TODO: Clean this puppy up, it's ugly
  """
  # number of sub interval in the grid
  nElts = len(grid) - 1
  # number of basis fcns; we lose them at the  ends
  n = nElts - 1
  F = _np.zeros((n, ))

  alpha = _np.sqrt(3./5.)
  w1 = 5./9.
  w2 = 8./9.
  w3 = 5./9.

  # Compute basis fcns at the quadrature points
  phi1 = 0.5 - alpha/2.
  phi2 = 0.5
  phi3 = 0.5 + alpha/2.

  # Loop over each subinterval. the first and last
  # subintervals are associated with a single basis
  # function. All others are associated with two
  # basis functions.
  for ii in range(0, nElts):
    l = grid[ii + 1] - grid [ii]
    x2 = (grid[ii] + grid[ii+1])/2.
    x1 = x2 - alpha*l/2.
    x3 = x2 + alpha*l/2.
    if ii < n:
      F[ii] =  F[ii] + 0.5*l*(
        w1*f(x1)*phi1 
        + w2*f(x2)*phi2 
        + w3*f(x3)*phi3
        )
    if ii > 0:
      F[ii-1] = F[ii-1] + 0.5*l*(
        w1*f(x1)*phi3
        + w2*f(x2)*phi2
        + w3*f(x3)*phi1
        )

  return F
