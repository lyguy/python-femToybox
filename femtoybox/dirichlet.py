import numpy as _np
import scipy.sparse as _sparse


def stiffnessEntry(grid, ii, jj):
  """Find \int \psi_i' \psi_j' dx where psi_i and \psi_j 
  are the respective entries of our pw-linear basis
  on the grid 'grid'

  TODO: * Make this more friendly to negative indexing,
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
    elif ii == L - 1:
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
  """
  n = len(grid) - 2
  diags = _np.zeros((3, n))
  offsets = [0, -1, 1]
  
  # Calculate the diagonal entries of the stiffness matrix
  for ii in range(n-1):
    diags[0, ii] = stiffnessEntry(grid, ii + 1, ii + 1)
    diags[1, ii] = stiffnessEntry(grid, ii + 1, ii + 2)
    diags[2, ii + 1] = diags[1, ii]
  diags[0, n - 1] = stiffnessEntry(grid, n, n)

  A = _sparse.dia_matrix((diags, offsets), shape=(n, n))

  # Scipy's solvers require a CRS or CSC formated sparse matrix.
  # Since the solvers are more efficient for CRS, we'll use it.
  A = A.tocsr()
  return A

def massEntry(grid, ii, jj):
  """Calculate \int psi_i psi_j dx, where psi_i is the ith
  pw-linear basis function
  """

  ii, jj = max(ii,jj), min(ii,jj)
  L = len(grid)
  if ii >= L:
    raise IndexError('index %i ' %(ii)
      + 'is out of bounds for grid with size %i' % (L))
 
  if jj < 0:
    raise IndexError('index %i ' %(jj)
      + 'is out of bounds for grid with size %i' %(L))

  # Integrals on either side of x_ii
  rightInt = lambda ii: (grid[ii + 1] - grid[ii]) / 6.0
  leftInt = lambda ii: (grid[ii] - grid[ii -1])/6.0

  def diag(ii):
    if ii == 0:
      return rightInt(ii)
    elif ii == L - 1:
      return leftInt(ii)
    else:
      return 2*(leftInt(ii) + rightInt(ii))

  options = { 0: diag, -1: rightInt, 1: leftInt}

  try:
    return options[ii-jj](ii)
  except KeyError:
    return 0.0


def zeroOrder(grid):
  """Form the mass matrix  C_{ij} = \int \phi_i+1 \phi_j+1 where 
  phi_i is the i'th basis function for the Dirichlet problem as described in the notes.

  TODO: * initialize C via coo, conversion to CRS is faster
        * similiar to the stiffness matrix, do this via a helper fcn 
  """
  n = len(grid) - 2
  diags = _np.zeros((3,n))
  offsets = [0, -1, 1]

  for ii in range(n-1):
    diags[0, ii] = massEntry(grid, ii+1, ii+1)
    diags[1, ii] = massEntry(grid, ii + 1, ii + 2)
    diags[2, ii+1] = diags[1, ii]
  diags[0, n-1] = massEntry(grid, n, n)

  C = _sparse.dia_matrix((diags, offsets), shape=(n,n))

  # SciPy's solvers need a CRS or CRC sparse matrix
  C = C.tocsr()
  return C


def advectEntry(grid, ii, jj):
  """
  Calculate \int \phi_i'\psi_j dx
  """
  L = len(grid)
  
  if ii >= L or jj >= L:
    raise IndexError('index is out of bounds for grid with size %i' %(L))
  if jj < 0 or ii < 0:
    raise IndexError('index is out of bounds for grid with size %i' %(L))

  options = { 0: 0.0, -1: -0.5, 1: 0.5}
  
  try:
    return options[jj-ii]*(grid[jj] - grid[ii])
  except KeyError:
    return 0.0

def advection(grid):
  """
  form the matrix for advection B{ij} = \int \phi_i' \phi_j
  """
  n = len(grid) - 2
  diags = _np.zeros((2, n))
  offsets = [-1, 1]

  for ii in range(n-1):
    diags[0, ii] = advectEntry(grid, ii+1, ii+1)
    diags[1, ii +1] = advectEntry(grid, ii+1, ii+2)

  B = _sparse.dia_matrix((diags, offsets), shape=(n, n))

  return B.tocsr()
  


def intF(f, grid):
  """
  Estimate \int f \phi_i using Gaussian quadrature

  TODO: Clean this puppy up, it's ugly
        * do this integration via builtin quadrature methods
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
