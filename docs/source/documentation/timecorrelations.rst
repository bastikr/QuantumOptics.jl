.. _section-timecorrelationfunctions:

Two-time correlation functions
==============================

.. jl:autofunction:: timecorrelations.jl correlation

.. jl:autofunction:: timecorrelations.jl spectrum

.. jl:autofunction:: timecorrelations.jl correlation2spectrum

As a brief example, say we want to calculate the two-time correlation function of a cavity field, i.e. :math:`g(t) = \langle a(t) a^\dagger(0)`.
We can do this with::

  basis = FockBasis(10)
  a = destroy(basis)
  κ = 1.0
  η = 0.3
  H = η*(a + dagger(a))
  J = [sqrt(2κ)*a]

  tspan = [0:0.1:10;]
  ρ0 = fockstate(basis, 0) ⊗ dagger(fockstate(basis, 0))
  g = timecorrelations.correlation(tspan, ρ0, H, J, dagger(a), a)

If we omit the list of times ``tspan``, the function automatically calculates the correlation until steady-state is reached::

  t_s, g_s = timecorrelations.correlation(ρ0, H, J, dagger(a), a)

To calculate the spectrum, we can either use the :jl:func:`timecorrelations.correlation2spectrum` to calculate it from a given correlation function, or alternatively, one can use :jl:func:`timecorrelations.spectrum`, which calculates the correlation function internally and returns the
Fourier transform.
