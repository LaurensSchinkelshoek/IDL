"""
Functions to operate on polynomials.
"""
__all__ = ['poly', 'roots', 'polyint', 'polyder', 'polyadd',
           'polysub', 'polymul', 'polydiv', 'polyval', 'poly1d',
           'polyfit', 'RankWarning']

import functools
import re
import warnings
import numpy.core.numeric as NX

from numpy.core import (isscalar, abs, finfo, atleast_1d, hstack, dot, array,
                        ones)
from numpy.core import overrides
from numpy.core.overrides import set_module
from numpy.lib.twodim_base import diag, vander
from numpy.lib.function_base import trim_zeros
from numpy.lib.type_check import iscomplex, real, imag, mintypecode
from numpy.linalg import eigvals, lstsq, inv

'''
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


@set_module('numpy')
class RankWarning(UserWarning):
    """
    Issued by `polyfit` when the Vandermonde matrix is rank deficient.
    For more information, a way to suppress the warning, and an example of
    `RankWarning` being issued, see `polyfit`.
    """
    pass


def _poly_dispatcher(seq_of_zeros):
    return seq_of_zeros


@array_function_dispatch(_poly_dispatcher)
def poly(seq_of_zeros):
    """
    Find the coefficients of a polynomial with the given sequence of roots.
    Returns the coefficients of the polynomial whose leading coefficient
    is one for the given sequence of zeros (multiple roots must be included
    in the sequence as many times as their multiplicity; see Examples).
    A square matrix (or array, which will be treated as a matrix) can also
    be given, in which case the coefficients of the characteristic polynomial
    of the matrix are returned.
    Parameters
    ----------
    seq_of_zeros : array_like, shape (N,) or (N, N)
        A sequence of polynomial roots, or a square array or matrix object.
    Returns
    -------
    c : ndarray
        1D array of polynomial coefficients from highest to lowest degree:
        ``c[0] * x**(N) + c[1] * x**(N-1) + ... + c[N-1] * x + c[N]``
        where c[0] always equals 1.
    Raises
    ------
    ValueError
        If input is the wrong shape (the input must be a 1-D or square
        2-D array).
    See Also
    --------
    polyval : Compute polynomial values.
    roots : Return the roots of a polynomial.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.
    Notes
    -----
    Specifying the roots of a polynomial still leaves one degree of
    freedom, typically represented by an undetermined leading
    coefficient. [1]_ In the case of this function, that coefficient -
    the first one in the returned array - is always taken as one. (If
    for some reason you have one other point, the only automatic way
    presently to leverage that information is to use ``polyfit``.)
    The characteristic polynomial, :math:`p_a(t)`, of an `n`-by-`n`
    matrix **A** is given by
        :math:`p_a(t) = \\mathrm{det}(t\\, \\mathbf{I} - \\mathbf{A})`,
    where **I** is the `n`-by-`n` identity matrix. [2]_
    References
    ----------
    .. [1] M. Sullivan and M. Sullivan, III, "Algebra and Trignometry,
       Enhanced With Graphing Utilities," Prentice-Hall, pg. 318, 1996.
    .. [2] G. Strang, "Linear Algebra and Its Applications, 2nd Edition,"
       Academic Press, pg. 182, 1980.
    Examples
    --------
    Given a sequence of a polynomial's zeros:
    >>> np.poly((0, 0, 0)) # Multiple root example
    array([1., 0., 0., 0.])
    The line above represents z**3 + 0*z**2 + 0*z + 0.
    >>> np.poly((-1./2, 0, 1./2))
    array([ 1.  ,  0.  , -0.25,  0.  ])
    The line above represents z**3 - z/4
    >>> np.poly((np.random.random(1)[0], 0, np.random.random(1)[0]))
    array([ 1.        , -0.77086955,  0.08618131,  0.        ]) # random
    Given a square array object:
    >>> P = np.array([[0, 1./3], [-1./2, 0]])
    >>> np.poly(P)
    array([1.        , 0.        , 0.16666667])
    Note how in all cases the leading coefficient is always 1.
    """
    seq_of_zeros = atleast_1d(seq_of_zeros)
    sh = seq_of_zeros.shape

    if len(sh) == 2 and sh[0] == sh[1] and sh[0] != 0:
        seq_of_zeros = eigvals(seq_of_zeros)
    elif len(sh) == 1:
        dt = seq_of_zeros.dtype
        # Let object arrays slip through, e.g. for arbitrary precision
        if dt != object:
            seq_of_zeros = seq_of_zeros.astype(mintypecode(dt.char))
    else:
        raise ValueError("input must be 1d or non-empty square 2d array.")

    if len(seq_of_zeros) == 0:
        return 1.0
    dt = seq_of_zeros.dtype
    a = ones((1,), dtype=dt)
    for k in range(len(seq_of_zeros)):
        a = NX.convolve(a, array([1, -seq_of_zeros[k]], dtype=dt),
                        mode='full')

    if issubclass(a.dtype.type, NX.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = NX.asarray(seq_of_zeros, complex)
        if NX.all(NX.sort(roots) == NX.sort(roots.conjugate())):
            a = a.real.copy()

    return a


def _roots_dispatcher(p):
    return p


@array_function_dispatch(_roots_dispatcher)
def roots(p):
    """
    Return the roots of a polynomial with coefficients given in p.
    The values in the rank-1 array `p` are coefficients of a polynomial.
    If the length of `p` is n+1 then the polynomial is described by::
      p[0] * x**n + p[1] * x**(n-1) + ... + p[n-1]*x + p[n]
    Parameters
    ----------
    p : array_like
        Rank-1 array of polynomial coefficients.
    Returns
    -------
    out : ndarray
        An array containing the roots of the polynomial.
    Raises
    ------
    ValueError
        When `p` cannot be converted to a rank-1 array.
    See also
    --------
    poly : Find the coefficients of a polynomial with a given sequence
           of roots.
    polyval : Compute polynomial values.
    polyfit : Least squares polynomial fit.
    poly1d : A one-dimensional polynomial class.
    Notes
    -----
    The algorithm relies on computing the eigenvalues of the
    companion matrix [1]_.
    References
    ----------
    .. [1] R. A. Horn & C. R. Johnson, *Matrix Analysis*.  Cambridge, UK:
        Cambridge University Press, 1999, pp. 146-7.
    Examples
    --------
    >>> coeff = [3.2, 2, 1]
    >>> np.roots(coeff)
    array([-0.3125+0.46351241j, -0.3125-0.46351241j])
    """
    # If input is scalar, this makes it an array
    p = atleast_1d(p)
    if p.ndim != 1:
        raise ValueError("Input must be a rank-1 array.")

    # find non-zero array entries
    non_zero = NX.nonzero(NX.ravel(p))[0]

    # Return an empty array if polynomial is all zeros
    if len(non_zero) == 0:
        return NX.array([])

    # find the number of trailing zeros -- this is the number of roots at 0.
    trailing_zeros = len(p) - non_zero[-1] - 1

    # strip leading and trailing zeros
    p = p[int(non_zero[0]):int(non_zero[-1])+1]

    # casting: if incoming array isn't floating point, make it floating point.
    if not issubclass(p.dtype.type, (NX.floating, NX.complexfloating)):
        p = p.astype(float)

    N = len(p)
    if N > 1:
        # build companion matrix and find its eigenvalues (the roots)
        A = diag(NX.ones((N-2,), p.dtype), -1)
        A[0,:] = -p[1:] / p[0]
        roots = eigvals(A)
    else:
        roots = NX.array([])

    # tack any zeros onto the back of the array
    roots = hstack((roots, NX.zeros(trailing_zeros, roots.dtype)))
    return roots


def _polyint_dispatcher(p, m=None, k=None):
    return (p,)


@array_function_dispatch(_polyint_dispatcher)
def polyint(p, m=1, k=None):
    """
    Return an antiderivative (indefinite integral) of a polynomial.
    The returned order `m` antiderivative `P` of polynomial `p` satisfies
    :math:`\\frac{d^m}{dx^m}P(x) = p(x)` and is defined up to `m - 1`
    integration constants `k`. The constants determine the low-order
    polynomial part
    .. math:: \\frac{k_{m-1}}{0!} x^0 + \\ldots + \\frac{k_0}{(m-1)!}x^{m-1}
    of `P` so that :math:`P^{(j)}(0) = k_{m-j-1}`.
    Parameters
    ----------
    p : array_like or poly1d
        Polynomial to integrate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of the antiderivative. (Default: 1)
    k : list of `m` scalars or scalar, optional
        Integration constants. They are given in the order of integration:
        those corresponding to highest-order terms come first.
        If ``None`` (default), all constants are assumed to be zero.
        If `m = 1`, a single scalar can be given instead of a list.
    See Also
    --------
    polyder : derivative of a polynomial
    poly1d.integ : equivalent method
    Examples
    --------
    The defining property of the antiderivative:
    >>> p = np.poly1d([1,1,1])
    >>> P = np.polyint(p)
    >>> P
     poly1d([ 0.33333333,  0.5       ,  1.        ,  0.        ]) # may vary
    >>> np.polyder(P) == p
    True
    The integration constants default to zero, but can be specified:
    >>> P = np.polyint(p, 3)
    >>> P(0)
    0.0
    >>> np.polyder(P)(0)
    0.0
    >>> np.polyder(P, 2)(0)
    0.0
    >>> P = np.polyint(p, 3, k=[6,5,3])
    >>> P
    poly1d([ 0.01666667,  0.04166667,  0.16666667,  3. ,  5. ,  3. ]) # may vary
    Note that 3 = 6 / 2!, and that the constants are given in the order of
    integrations. Constant of the highest-order polynomial term comes first:
    >>> np.polyder(P, 2)(0)
    6.0
    >>> np.polyder(P, 1)(0)
    5.0
    >>> P(0)
    3.0
    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of integral must be positive (see polyder)")
    if k is None:
        k = NX.zeros(m, float)
    k = atleast_1d(k)
    if len(k) == 1 and m > 1:
        k = k[0]*NX.ones(m, float)
    if len(k) < m:
        raise ValueError(
              "k must be a scalar or a rank-1 array of length 1 or >m.")

    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    if m == 0:
        if truepoly:
            return poly1d(p)
        return p
    else:
        # Note: this must work also with object and integer arrays
        y = NX.concatenate((p.__truediv__(NX.arange(len(p), 0, -1)), [k[0]]))
        val = polyint(y, m - 1, k=k[1:])
        if truepoly:
            return poly1d(val)
        return val


def _polyder_dispatcher(p, m=None):
    return (p,)


@array_function_dispatch(_polyder_dispatcher)
def polyder(p, m=1):
    """
    Return the derivative of the specified order of a polynomial.
    Parameters
    ----------
    p : poly1d or sequence
        Polynomial to differentiate.
        A sequence is interpreted as polynomial coefficients, see `poly1d`.
    m : int, optional
        Order of differentiation (default: 1)
    Returns
    -------
    der : poly1d
        A new polynomial representing the derivative.
    See Also
    --------
    polyint : Anti-derivative of a polynomial.
    poly1d : Class for one-dimensional polynomials.
    Examples
    --------
    The derivative of the polynomial :math:`x^3 + x^2 + x^1 + 1` is:
    >>> p = np.poly1d([1,1,1,1])
    >>> p2 = np.polyder(p)
    >>> p2
    poly1d([3, 2, 1])
    which evaluates to:
    >>> p2(2.)
    17.0
    We can verify this, approximating the derivative with
    ``(f(x + h) - f(x))/h``:
    >>> (p(2. + 0.001) - p(2.)) / 0.001
    17.007000999997857
    The fourth-order derivative of a 3rd-order polynomial is zero:
    >>> np.polyder(p, 2)
    poly1d([6, 2])
    >>> np.polyder(p, 3)
    poly1d([6])
    >>> np.polyder(p, 4)
    poly1d([0.])
    """
    m = int(m)
    if m < 0:
        raise ValueError("Order of derivative must be positive (see polyint)")

    truepoly = isinstance(p, poly1d)
    p = NX.asarray(p)
    n = len(p) - 1
    y = p[:-1] * NX.arange(n, 0, -1)
    if m == 0:
        val = p
    else:
        val = polyder(y, m - 1)
    if truepoly:
        val = poly1d(val)
    return val


def _polyfit_dispatcher(x, y, deg, rcond=None, full=None, w=None, cov=None):
    return (x, y, w)

'''
#@array_function_dispatch(_polyfit_dispatcher)
def polyfit_(x, y, deg, alpha , rcond=None, full=False, w=None, cov=False):
    """
    Least squares polynomial fit.
    Fit a polynomial ``p(x) = p[0] * x**deg + ... + p[deg]`` of degree `deg`
    to points `(x, y)`. Returns a vector of coefficients `p` that minimises
    the squared error in the order `deg`, `deg-1`, ... `0`.
    The `Polynomial.fit <numpy.polynomial.polynomial.Polynomial.fit>` class
    method is recommended for new code as it is more stable numerically. See
    the documentation of the method for more information.
    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int
        Degree of the fitting polynomial
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (M,), optional
        Weights to apply to the y-coordinates of the sample points. For
        gaussian uncertainties, use 1/sigma (not 1/sigma**2).
    cov : bool or str, optional
        If given and not `False`, return not just the estimate but also its
        covariance matrix. By default, the covariance are scaled by
        chi2/sqrt(N-dof), i.e., the weights are presumed to be unreliable
        except in a relative sense and everything is scaled such that the
        reduced chi2 is unity. This scaling is omitted if ``cov='unscaled'``,
        as is relevant for the case that the weights are 1/sigma**2, with
        sigma known to be a reliable estimate of the uncertainty.
    Returns
    -------
    p : ndarray, shape (deg + 1,) or (deg + 1, K)
        Polynomial coefficients, highest power first.  If `y` was 2-D, the
        coefficients for `k`-th data set are in ``p[:,k]``.
    residuals, rank, singular_values, rcond
        Present only if `full` = True.  Residuals is sum of squared residuals
        of the least-squares fit, the effective rank of the scaled Vandermonde
        coefficient matrix, its singular values, and the specified value of
        `rcond`. For more details, see `linalg.lstsq`.
    V : ndarray, shape (M,M) or (M,M,K)
        Present only if `full` = False and `cov`=True.  The covariance
        matrix of the polynomial coefficient estimates.  The diagonal of
        this matrix are the variance estimates for each coefficient.  If y
        is a 2-D array, then the covariance matrix for the `k`-th data set
        are in ``V[:,:,k]``
    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if `full` = False.
        The warnings can be turned off by
        >>> import warnings
        >>> warnings.simplefilter('ignore', np.RankWarning)
    See Also
    --------
    polyval : Compute polynomial values.
    linalg.lstsq : Computes a least-squares fit.
    scipy.interpolate.UnivariateSpline : Computes spline fits.
    Notes
    -----
    The solution minimizes the squared error
    .. math ::
        E = \\sum_{j=0}^k |p(x_j) - y_j|^2
    in the equations::
        x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
        x[1]**n * p[0] + ... + x[1] * p[n-1] + p[n] = y[1]
        ...
        x[k]**n * p[0] + ... + x[k] * p[n-1] + p[n] = y[k]
    The coefficient matrix of the coefficients `p` is a Vandermonde matrix.
    `polyfit` issues a `RankWarning` when the least-squares fit is badly
    conditioned. This implies that the best fit is not well-defined due
    to numerical error. The results may be improved by lowering the polynomial
    degree or by replacing `x` by `x` - `x`.mean(). The `rcond` parameter
    can also be set to a value smaller than its default, but the resulting
    fit may be spurious: including contributions from the small singular
    values can add numerical noise to the result.
    Note that fitting polynomial coefficients is inherently badly conditioned
    when the degree of the polynomial is large or the interval of sample points
    is badly centered. The quality of the fit should always be checked in these
    cases. When polynomial fits are not satisfactory, splines may be a good
    alternative.
    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting
    .. [2] Wikipedia, "Polynomial interpolation",
           https://en.wikipedia.org/wiki/Polynomial_interpolation
    Examples
    --------
    >>> import warnings
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> z = np.polyfit(x, y, 3)
    >>> z
    array([ 0.08703704, -0.81349206,  1.69312169, -0.03968254]) # may vary
    It is convenient to use `poly1d` objects for dealing with polynomials:
    >>> p = np.poly1d(z)
    >>> p(0.5)
    0.6143849206349179 # may vary
    >>> p(3.5)
    -0.34732142857143039 # may vary
    >>> p(10)
    22.579365079365115 # may vary
    High-order polynomials may oscillate wildly:
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter('ignore', np.RankWarning)
    ...     p30 = np.poly1d(np.polyfit(x, y, 30))
    ...
    >>> p30(4)
    -0.80000000000000204 # may vary
    >>> p30(5)
    -0.99999999999999445 # may vary
    >>> p30(4.5)
    -0.10547061179440398 # may vary
    Illustration:
    >>> import matplotlib.pyplot as plt
    >>> xp = np.linspace(-2, 6, 100)
    >>> _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
    >>> plt.ylim(-2,2)
    (-2, 2)
    >>> plt.show()
    """
    order = int(deg) + 1
    x = NX.asarray(x) + 0.0
    y = NX.asarray(y) + 0.0

    # check arguments.
    if deg < 0:
        raise ValueError("expected deg >= 0")
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if y.ndim < 1 or y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.shape[0] != y.shape[0]:
        raise TypeError("expected x and y to have same length")

    # set rcond
    if rcond is None:
        rcond = len(x)*finfo(x.dtype).eps

    # set up least squares equation for powers of x
    lhs = vander(x, order)
    rhs = y

    # apply weighting
    if w is not None:
        w = NX.asarray(w) + 0.0
        if w.ndim != 1:
            raise TypeError("expected a 1-d array for weights")
        if w.shape[0] != y.shape[0]:
            raise TypeError("expected w and y to have the same length")
        lhs *= w[:, NX.newaxis]
        if rhs.ndim == 2:
            rhs *= w[:, NX.newaxis]
        else:
            rhs *= w

    # scale lhs to improve condition number and solve
    scale = NX.sqrt((lhs*lhs).sum(axis=0))+alpha*(lhs*lhs).sum(axis=0)
    lhs /= scale
    c, resids, rank, s = lstsq(lhs, rhs, rcond)
    c = (c.T/scale).T  # broadcast scale coefficients

    # warn on rank reduction, which indicates an ill conditioned matrix
    if rank != order and not full:
        msg = "Polyfit may be poorly conditioned"
        warnings.warn(msg, RankWarning, stacklevel=4)

    if full:
        return c, resids, rank, s, rcond
    elif cov:
        Vbase = inv(dot(lhs.T, lhs))
        Vbase /= NX.outer(scale, scale)
        if cov == "unscaled":
            fac = 1
        else:
            if len(x) <= order:
                raise ValueError("the number of data points must exceed order "
                                 "to scale the covariance matrix")
            # note, this used to be: fac = resids / (len(x) - order - 2.0)
            # it was deciced that the "- 2" (originally justified by "Bayesian
            # uncertainty analysis") is not was the user expects
            # (see gh-11196 and gh-11197)
            fac = resids / (len(x) - order)
        if y.ndim == 1:
            return c, Vbase * fac
        else:
            return c, Vbase[:,:, NX.newaxis] * fac
    else:
        return c