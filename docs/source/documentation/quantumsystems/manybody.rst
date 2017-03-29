.. _section-manybody:

Many-body system
================

Describing systems consisting of many identical particles in a tensor product space created out of single particle Hilbert spaces leads to the problem that not all states in this space correspond to real physical states. In this picture one would have to restrict the Hilbert space to a subspace that is invariant under permutation of particles. However, it is also possible to find a valid description that doesn't first introduce redundant states that later on have to be eliminated. The general idea is to choose an arbitrary basis :math:`\{\left|u_i\right\rangle\}_i` of the single particle Hilbert space and create the N-particle Hilbert space from states that count how many particles are in each of these states - which will in the following be denoted as :math:`\left|\{n\}\right\rangle`. Of course the sum of these occupation numbers has to be identical to the number of particles. For fermionic particles an additional restriction is that there can't be more than one particle in one state.

This concept is captured in the :jl:type:`ManyBodyBasis` type.

Connection between additive single particle operator :math:`\sum_i x_i` and its corresponding many-body operator:

.. math::

    X = \sum_{ij} a_i^\dagger a_j
                    \left\langle u_i \right|
                    x
                    \left| u_j \right\rangle

Connection between additive two particle operator :math:`\sum_{i \neq j} V_{ij}` and its corresponding many-body operator:

.. math::

    X = \sum_{ijkl} a_i^\dagger a_j^\dagger a_k a_l
            \left\langle u_i \right| \left\langle u_j \right|
            x
            \left| u_k \right\rangle \left| u_l \right\rangle

The creation of the many-body operators is implemented with the :jl:func:`manybodyoperator` function.
