from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

import jax
import jax.numpy as jnp
import lineax as lx
from equinox.internal import ω

from .._custom_types import Args, BoolScalarLike, DenseInfo, RealScalarLike, VF, Y
from .._local_interpolation import LocalLinearInterpolation
from .._solution import RESULTS
from .._term import AbstractTerm
from .base import AbstractAdaptiveSolver
import numpy as np
import jax.tree_util as jtu
import lineax.internal as lxi
import jax.lax as lax


_SolverState: TypeAlias = None


@dataclass(frozen=True)
class _RosenbrockTableau:
    """The coefficient tableau for Rosenbrock methods"""

    m_sol: np.ndarray
    m_error: np.ndarray

    a_lower: tuple[np.ndarray, ...]
    c_lower: tuple[np.ndarray, ...]

    α: np.ndarray
    γ: np.ndarray

    num_stages: int

    # Example tableau
    #
    # α1 | a11 a12 a13 | c11 c12 c13 | γ1
    # α1 | a21 a22 a23 | c21 c22 c23 | γ2
    # α3 | a31 a32 a33 | c31 c32 c33 | γ3
    # ---+----------------
    #    | m1  m2  m3
    #    | me1 me2 me3


_tableau = _RosenbrockTableau(
    m_sol=np.array([2.0, 0.5773502691896258, 0.4226497308103742]),
    m_error=np.array([2.113248654051871, 1.0, 0.4226497308103742]),
    a_lower=(np.array([1.267949192431123, 0.0]), np.array([1.267949192431123, 0.0])),
    c_lower=(
        np.array([-1.607695154586736, 0.0]),
        np.array([-3.464101615137755, -1.732050807568877]),
    ),
    α=np.array([0.0, 1.0, 1.0]),
    γ=np.array(
        [
            0.7886751345948129,
            -0.2113248654051871,
            -1.0773502691896260,
        ]
    ),
    num_stages=3,
)


class Ros3p(AbstractAdaptiveSolver):
    r"""Ros3p method.

    3rd order Rosenbrock method for solving stiff equation. Uses a 1st order local linear
    interpolation for dense output.

    ??? cite "Reference"

        ```bibtex
        @article{LangVerwer2001ROS3P,
          author    = {Lang, J. and Verwer, J.},
          title     = {ROS3P---An Accurate Third-Order Rosenbrock Solver Designed
                       for Parabolic Problems},
          journal   = {BIT Numerical Mathematics},
          volume    = {41},
          number    = {4},
          pages     = {731--738},
          year      = {2001},
          doi       = {10.1023/A:1021900219772}
         }
         ```
    """

    term_structure: ClassVar = AbstractTerm
    interpolation_cls: ClassVar[Callable[..., LocalLinearInterpolation]] = (
        LocalLinearInterpolation
    )

    tableau: ClassVar[_RosenbrockTableau] = _tableau

    def init(self, terms, t0, t1, y0, args) -> _SolverState:
        del terms, t0, t1, y0, args
        return None

    def order(self, terms):
        return 3

    def step(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple[Y, Y, DenseInfo, _SolverState, RESULTS]:
        del made_jump, solver_state

        time_derivative = jax.jacfwd(lambda t: terms.vf(t, y0, args))(t0)
        control = terms.contr(t0, t1)

        γ = jnp.array(self.tableau.γ)
        α = jnp.array(self.tableau.α)
        a_lower = jnp.array(self.tableau.a_lower)
        c_lower = jnp.array(self.tableau.c_lower)
        m_sol = jnp.array(self.tableau.m_sol)
        m_error = jnp.array(self.tableau.m_error)

        # common L.H.S
        eye_shape = jax.ShapeDtypeStruct(
            (time_derivative.shape[-1],), time_derivative.dtype
        )
        A = (lx.IdentityLinearOperator(eye_shape) / (control * γ[0])) - (
            lx.JacobianLinearOperator(
                lambda y, args: terms.vf(t0, y, args), y0, args=args
            )
        )
        
        u = jnp.zeros((len(time_derivative),self.tableau.num_stages))
        
        # stage 1
        stage_1_b = (
            terms.vf(
                (t0**ω + (α[0] ** ω * control**ω)).ω,
                y0,
                args,
            )
            ** ω
            + (control**ω * γ[0] ** ω * time_derivative**ω)
        ).ω

        # solving Ax=b
        u1 = lx.linear_solve(A, stage_1_b).value

        # stage 2
        stage_2_b = (
            terms.vf(
                (t0**ω + (α[1] ** ω * control**ω)).ω,
                (y0**ω + (a_lower[0][0] ** ω * u1**ω)).ω,
                args,
            )
            ** ω
            + ((c_lower[0][0] ** ω / control**ω) * u1**ω)
            + (control**ω * γ[1] ** ω * time_derivative**ω)
        ).ω

        # solving Ax=b
        u2 = lx.linear_solve(A, stage_2_b).value

        # stage 3
        stage_3_b = (
            terms.vf(
                (t0**ω + α[2] ** ω * control**ω).ω,
                (y0**ω + (a_lower[1][0] ** ω * u1**ω) + (a_lower[1][1] ** ω * u2**ω)).ω,
                args,
            )
            ** ω
            + ((c_lower[1][0] ** ω / control**ω) * u1**ω)
            + ((c_lower[1][1] ** ω / control**ω) * u2**ω)
            + (control**ω * γ[2] ** ω * time_derivative**ω)
        ).ω

        # solving Ax=b
        u3 = lx.linear_solve(A, stage_3_b).value

        y1 = (
            y0**ω
            + m_sol[0] ** ω * u1**ω
            + m_sol[1] ** ω * u2**ω
            + m_sol[2] ** ω * u3**ω
        ).ω
        y1_lower = (
            y0**ω
            + m_error[0] ** ω * u1**ω
            + m_error[1] ** ω * u2**ω
            + m_error[2] ** ω * u3**ω
        ).ω

        y1_error = y1 - y1_lower
        dense_info = dict(y0=y0, y1=y1)
        return y1, y1_error, dense_info, None, RESULTS.successful

    def func(
        self,
        terms: AbstractTerm,
        t0: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> VF:
        return terms.vf(t0, y0, args)


Ros3p.__init__.__doc__ = """**Arguments:** None"""
