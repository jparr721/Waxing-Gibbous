import os
from typing import Final, List, Union

import numpy as np
from loguru import logger
from tqdm import tqdm

if "NO_JIT" in os.environ:
    from mpm.mpm_nonjit import apply_boundary_conditions, g2p, grid_op, p2g
else:
    from mpm.mpm import apply_boundary_conditions, g2p, grid_op, p2g


class Engine(object):
    def __init__(self, outdir="tmp"):
        # How-swappable functor types
        self.p2g = p2g
        self.g2p = g2p
        self.grid_op = grid_op
        self.apply_boundary_conditions = apply_boundary_conditions

        # Final sim constants
        self.grid_res: Final[int] = 64
        self.dt: Final[float] = 1e-4
        self.dx: Final[float] = 1 / self.grid_res
        self.inv_dx: Final[float] = 1 / self.dx
        self.mass: Final[float] = 1.0
        self.vol: Final[float] = 1.0

    def simulate(
        self,
        x: np.ndarray,
        *,
        use_gui=True,
        model="jelly",
        boundary_ops="sticky",
        steps=2000,
        gravity=-300.0,
        E=1e3,
        nu=0.2,
        coarsening=1,
    ):
        """Runs a simulation provided an input list of starting points

        Args:
            x (np.ndarray): The starting points
            use_gui (bool, optional): Whether or not to use the gui. Defaults to True.
            model (str, optional): The model to use. Defaults to "jelly".
            boundary_ops (str, optional): The boundary conditions. Defaults to "sticky".
            steps (int, optional): The number of sim steps. Defaults to 2000.
            gravity (float, optional): How much gravity to apply. Defaults to -300.0.
            E (_type_, optional): The youngs modulus. Defaults to 1e3.
            nu (float, optional): The poissons ratio. Defaults to 0.2.
            coarsening (int, optional): Sampling frequency for datasets. Defaults to 1.
        """
        self.x = x
        _n = len(self.x)
        dim = 2
        self.v = np.zeros((_n, dim), dtype=np.float64)
        self.F = np.array([np.eye(dim, dtype=np.float64) for _ in range(_n)])
        self.C = np.zeros((_n, dim, dim), dtype=np.float64)
        self.Jp = np.ones((_n, 1), dtype=np.float64)
        self.mu_0 = E / (2 * (1 + nu))
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

        self.historical_positions = []
        self.initial_grid_state = []
        self.final_grid_state = []

        self.current_step = 0
        try:
            self._run_sim(model, use_gui, steps, coarsening, gravity, boundary_ops)
        except Exception as e:
            logger.error(f"Sim crashed: {e}")
            raise e

        if use_gui:
            self._show_gui()

    def _run_sim(
        self,
        model: Union[str, List[str]],
        use_gui: bool,
        steps: int,
        coarsening: int,
        gravity: float,
        boundary_ops: str,
    ):
        for self.current_step in tqdm(range(steps)):
            self.gm = np.zeros((self.grid_res, self.grid_res, 1))
            self.gv = np.zeros((self.grid_res, self.grid_res, 2))

            self.p2g(
                self.inv_dx,
                self.mu_0,
                self.lambda_0,
                self.mass,
                self.dx,
                self.dt,
                self.vol,
                self.gv,
                self.gm,
                self.x,
                self.v,
                self.F,
                self.C,
                self.Jp,
                model,
            )

            if self.current_step % coarsening == 0:
                self.initial_grid_state.append(self.gv.copy())

            self.grid_op(self.dx, self.dt, gravity, self.gv, self.gm)
            self.apply_boundary_conditions(self.gv, boundary_ops)

            if self.current_step % coarsening == 0:
                self.final_grid_state.append(self.gv.copy())

            self.g2p(
                self.inv_dx,
                self.dt,
                self.gv,
                self.x,
                self.v,
                self.F,
                self.C,
                self.Jp,
                model,
            )

            if use_gui:
                self.historical_positions.append(self.x.copy())

    def generate(
        self,
        x: np.ndarray,
        model: np.ndarray,
        steps: int = 2000,
        gmin: int = -300,
        gmax: int = -10,
        incr: int = 10,
        coarsening: int = 10,
    ):
        initial = []
        final = []

        if gmin == gmax:
            gmax += 1

        for gravity in tqdm(range(gmin, gmax, incr)):
            self.simulate(
                x.copy(),
                use_gui=False,
                model=model,
                steps=steps,
                gravity=gravity,
                coarsening=coarsening,
            )

            assert len(self.initial_grid_state) == len(self.final_grid_state)
            initial.extend(self.initial_grid_state)
            final.extend(self.final_grid_state)

        initial = np.stack(initial)
        final = np.stack(final)

        logger.debug(f"initial.shape: {initial.shape}")
        logger.debug(f"final.shape: {final.shape}")

        if len(set(model)) <= 1:
            m = model[0]
        else:
            m = "all"

        np.savez(
            f"dataset_res_{x.shape[0]}_gmin_{gmin}_to_gmax_{gmax}_coarse_{coarsening}_{m}.npz",
            initial=initial,
            final=final,
        )

    def _show_gui(self):
        import taichi as ti

        ti.init(arch=ti.gpu)
        gui = ti.GUI()
        while gui.running and not gui.get_event(gui.ESCAPE):
            for i in range(0, len(self.historical_positions), 10):
                gui.clear(0x112F41)
                gui.rect(
                    np.array((0.04, 0.04)),
                    np.array((0.96, 0.96)),
                    radius=2,
                    color=0x4FB99F,
                )
                gui.circles(self.historical_positions[i], radius=1.5, color=0xED553B)
                gui.show()
