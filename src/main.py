import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import taichi as ti
import typer
from loguru import logger
from rich.console import Console
from rich.progress import track
from rich.table import Table
from rich.traceback import install
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

from network import make_model, plot, save_model, train_model
from praxis import *

# Set up rich to MITM all errors
install()

app = typer.Typer()

ti.init(arch=ti.metal)

# Flag to activate the fluid solver
use_fluid_solver = False

# Set the grid padding on the boundary
PADDING_AMOUNT = 3
GRID_DIMENSIONS = 64

if use_fluid_solver:
    beam_starting_x = 3
    beam_starting_y = 3
    beam_width = 20
    beam_height = 6
else:
    beam_starting_x = 6
    beam_starting_y = 12
    beam_width = 20
    beam_height = 6

n_particles = beam_width * beam_height * 4

dx, inv_dx = 1 / float(GRID_DIMENSIONS), float(GRID_DIMENSIONS)
dt = 1e-4
gravity = 20.0

point_volume = dx * 0.5
point_density = 1
p_mass = point_volume * point_density

youngs_modulus = 1_000
poissons_ratio = 0.2

mu = youngs_modulus / (2 * (1 + poissons_ratio))
lambda_ = (
    youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))
)
positions = ti.Vector.field(2, dtype=float, shape=n_particles)
velocity = ti.Vector.field(2, dtype=float, shape=n_particles)
affine_velocity = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
deformation_gradient = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
color = ti.field(dtype=int, shape=n_particles)
plastic_deformation = ti.field(dtype=float, shape=n_particles)
grid_velocity = ti.Vector.field(2, dtype=float, shape=(GRID_DIMENSIONS, GRID_DIMENSIONS))
grid_mass = ti.field(dtype=float, shape=(GRID_DIMENSIONS, GRID_DIMENSIONS))
grid_index = ti.field(dtype=int, shape=(GRID_DIMENSIONS, GRID_DIMENSIONS))


@ti.kernel
def clean_grid():
    for i, j in grid_mass:
        grid_velocity[i, j] = [0, 0]
        grid_mass[i, j] = 0


@ti.kernel
def particle_to_grid():
    for p in positions:
        base = (positions[p] * inv_dx - 0.5).cast(int)
        fx = positions[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        deformation_gradient[p] = (
            ti.Matrix.identity(ti.f32, 2) + dt * affine_velocity[p]
        ) @ deformation_gradient[p]
        R, S = ti.polar_decompose(deformation_gradient[p])
        plastic_deformation[p] = plastic_deformation[p] * (
            1 + dt * affine_velocity[p].trace()
        )
        cauchy = (
            2 * mu * (deformation_gradient[p] - R)
            + lambda_
            * (
                R.transpose() @ deformation_gradient[p] - ti.Matrix.identity(ti.f32, 2)
            ).trace()
            * R
        ) @ deformation_gradient[p].transpose()
        stress = (-dt * point_volume * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * affine_velocity[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_velocity[base + offset] += weight * (
                p_mass * velocity[p] + affine @ dpos
            )
            grid_mass[base + offset] += weight * p_mass


@ti.kernel
def grid_update(g: ti.f32):
    for i, j in grid_mass:
        if grid_mass[i, j] > 0:
            grid_velocity[i, j] = (1 / grid_mass[i, j]) * grid_velocity[
                i, j
            ]  # Momentum to velocity
            grid_velocity[i, j][1] -= dt * g

        # Sticky boundary conditions
        if i < PADDING_AMOUNT and grid_velocity[i, j][0] < 0:
            grid_velocity[i, j][0] = 0
        if i > GRID_DIMENSIONS - PADDING_AMOUNT and grid_velocity[i, j][0] > 0:
            grid_velocity[i, j][0] = 0
        if j < PADDING_AMOUNT and grid_velocity[i, j][1] < 0:
            grid_velocity[i, j][1] = 0
        if j > GRID_DIMENSIONS - PADDING_AMOUNT and grid_velocity[i, j][1] > 0:
            grid_velocity[i, j][1] = 0

        if not use_fluid_solver:
            # Fixed left side of the beam for elastic sim
            if i < beam_starting_x + PADDING_AMOUNT:
                grid_velocity[i, j] = [0, 0]


@ti.kernel
def grid_to_particle():
    for p in positions:
        base = (positions[p] * inv_dx - 0.5).cast(int)
        fx = positions[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_velocity[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)

        # Advect the velocity and positions
        velocity[p], affine_velocity[p] = new_v, new_C
        positions[p] += dt * velocity[p]

        if use_fluid_solver:
            # Fluid update
            U, sig, V = ti.svd(deformation_gradient[p])
            J = 1.0
            for d in ti.static(range(2)):
                J *= sig[d, d]
                # Reset deformation gradient to avoid numerical instability
                # in fluid solvers
                deformation_gradient[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)


@ti.kernel
def initialize():
    for i in range(beam_width * 2):
        for j in range(beam_height * 2):
            idx = j + i * beam_height * 2
            positions[idx] = ti.Matrix(
                [
                    (beam_starting_x + i * 0.5 + ti.random()) * dx,
                    (beam_starting_y + j * 0.5 + ti.random()) * dx,
                ]
            )
            color[idx] = 1
            velocity[idx] = ti.Matrix([0, 0])
            deformation_gradient[idx] = ti.Matrix([[1, 0], [0, 1]])
            plastic_deformation[idx] = 1


@ti.kernel
def particle_to_grid_static():
    for p in positions:
        base = (positions[p] * inv_dx - 0.5).cast(int)
        fx = positions[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1]

            # Zero deformation static position always has zero velocity
            grid_velocity[base + offset].fill(0)
            grid_mass[base + offset] += weight * p_mass


def compute_active_dof_rows_and_cols():
    for i in range(GRID_DIMENSIONS):
        for j in range(GRID_DIMENSIONS):
            grid_index[i, j] = -1

    rn = 0
    for i in range(0, GRID_DIMENSIONS):
        for j in range(0, GRID_DIMENSIONS):
            if grid_mass[i, j] > 0 and i >= beam_starting_x + PADDING_AMOUNT:
                grid_index[i, j] = rn
                rn += 1

    cn = 0
    for i in range(0, n_particles):
        if positions[i][0] > (beam_starting_x + 1) * dx:
            cn += 1

    return rn, cn


def get_sparse_matrix_info_2D():
    arr_len = 0
    for p in range(n_particles):
        if positions[p][0] > (beam_starting_x + 1) * dx:
            basex = int(positions[p][0] * inv_dx - 0.5)
            basey = int(positions[p][1] * inv_dx - 0.5)
            for i in range(0, 3):
                for j in range(0, 3):
                    if (
                        grid_mass[basex + i, basey + j] > 0
                        and basex + i >= beam_starting_x + PADDING_AMOUNT
                    ):
                        arr_len += 1

    # The goal here is to build the linear system of all the subsystems for each particle
    # so that we can solve the least squares problem
    row = np.zeros(arr_len * 4)
    col = np.zeros(arr_len * 4)
    dat = np.zeros(arr_len * 4)
    arr_len = 0
    col_num = 0
    for p in range(n_particles):
        if positions[p][0] > (beam_starting_x + 1) * dx:
            basex = int(positions[p][0] * inv_dx - 0.5)
            basey = int(positions[p][1] * inv_dx - 0.5)
            fx = np.array(
                [
                    float(positions[p][0] * inv_dx - basex),
                    float(positions[p][1] * inv_dx - basey),
                ]
            )
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for i in range(0, 3):
                for j in range(0, 3):
                    offset = np.array([i, j]).astype(float)
                    dpos = (offset - fx) * dx
                    weight = w[i][0] * w[j][1]
                    if (
                        grid_mass[basex + i, basey + j] > 0
                        and basex + i >= beam_starting_x + PADDING_AMOUNT
                    ):
                        row_id = grid_index[basex + i, basey + j]
                        col_id = col_num
                        row[arr_len * 4 + 0] = row_id * 2 + 0
                        col[arr_len * 4 + 0] = col_id * 3 + 0
                        dat[arr_len * 4 + 0] = weight * dpos[0]

                        row[arr_len * 4 + 1] = row_id * 2 + 0
                        col[arr_len * 4 + 1] = col_id * 3 + 2
                        dat[arr_len * 4 + 1] = weight * dpos[1]

                        row[arr_len * 4 + 2] = row_id * 2 + 1
                        col[arr_len * 4 + 2] = col_id * 3 + 2
                        dat[arr_len * 4 + 2] = weight * dpos[0]

                        row[arr_len * 4 + 3] = row_id * 2 + 1
                        col[arr_len * 4 + 3] = col_id * 3 + 1
                        dat[arr_len * 4 + 3] = weight * dpos[1]
                        arr_len += 1
            col_num += 1
    return row, col, dat


def get_active_DOF_rest_forces(row_num):
    rhs = np.zeros(row_num * 2)
    row_num = 0
    for i in range(0, GRID_DIMENSIONS):
        for j in range(0, GRID_DIMENSIONS):
            if grid_mass[i, j] > 0 and i >= beam_starting_x + PADDING_AMOUNT:
                rhs[row_num * 2 + 0] = 0
                rhs[row_num * 2 + 1] = gravity * grid_mass[i, j]
                row_num += 1
    return rhs


def solve_least_squares(A_sp, b):
    c, _, _, _ = lsqr(A_sp, b)[:4]
    return c


def evaluate_affine_mapping(f00, f11, f10):
    localF = np.array([[f00, f10], [f10, f11]])
    U, S, Vt = np.linalg.svd(localF)
    R = U @ Vt
    S = Vt.T @ np.diag(S) @ Vt
    cauchy = (
        2 * mu * (localF - R) + lambda_ * np.trace(R.T @ localF - np.eye(2)) * R
    ) @ localF.T
    affine = (-point_volume * 4 * inv_dx * inv_dx) * cauchy
    return affine


def find_solution(A0):
    n = 3
    t0 = 1e-20
    h0 = 1
    prin = 0

    # Use praxis linear rootfinding alg to evaluate the
    # linear solution for the mapping between the forces and the deformation
    def f(r, n):
        A = evaluate_affine_mapping(r[0], r[1], r[2])
        diff = A - A0
        val = diff[0, 0] * diff[0, 0]
        val += diff[0, 1] * diff[0, 1]
        val += diff[1, 0] * diff[1, 0]
        val += diff[1, 1] * diff[1, 1]
        return val

    r = np.array([1.0, 1.0, 0.0])
    pr, r = praxis(t0, h0, n, prin, r, f)

    return r


@ti.kernel
def test_particle_to_grid_static():
    for p in positions:
        base = (positions[p] * inv_dx - 0.5).cast(int)
        fx = positions[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        affine = deformation_gradient[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_velocity[base + offset] += weight * (affine @ dpos)
            grid_mass[base + offset] += weight * p_mass


def verify_global_solver(c):
    clean_grid()
    col_num = 0
    for p in range(0, n_particles):
        if positions[p][0] >= (beam_starting_x + 1) * dx:
            deformation_gradient[p][0, 0] = c[col_num * 3 + 0]
            deformation_gradient[p][1, 1] = c[col_num * 3 + 1]
            deformation_gradient[p][0, 1] = c[col_num * 3 + 2]
            deformation_gradient[p][1, 0] = c[col_num * 3 + 2]
            col_num = col_num + 1

    test_particle_to_grid_static()

    # for i in range(grid_dimensions):
    #     for j in range(0, grid_dimensions):
    #         if grid_mass[i, j] > 0 and i >= beam_starting_x + PADDING_AMOUNT:
    #             if not np.isclose(grid_velocity[i, j][0], 0, atol=1e-6):
    #                 j}ob_progress.console.print(
    #                     "Error in global step: ", i, j, grid_velocity[i, j][0] - 0
    #                 )
    #             if not np.isclose(
    #                 grid_velocity[i, j][1], gravity * grid_mass[i, j], atol=1e-6
    #             ):
    #                 job_progress.console.print(
    #                     "Error in global step: ",
    #                     i,
    #                     j,
    #                     grid_velocity[i, j][1] - gravity * grid_mass[i, j],
    #                 )


def verify_local_solver(res_array, col_num, c):
    for i in range(col_num):
        affine = evaluate_affine_mapping(
            res_array[i * 3 + 0], res_array[i * 3 + 1], res_array[i * 3 + 2]
        )
        res_vec = np.array([affine[0, 0], affine[1, 1], affine[1, 0]])
        tar_vec = np.array([c[i * 3], c[i * 3 + 1], c[i * 3 + 2]])
        # if not np.allclose(res_vec, tar_vec, atol=1e-6):
        #     job_progress.console.print(
        #         "Error in local step: ",
        #         c[i * 3],
        #         c[i * 3 + 1],
        #         c[i * 3 + 2],
        #         res_vec,
        #         tar_vec,
        #         res_vec - tar_vec,
        #     )
        # job_progress.advance(task)


def nn_get_baseline_force_matrix():
    # 4-dimensional force vector field dimxdimx4
    # dim 1 and 2 are the uniaxial forces along the y axis
    # dim 3 is the mass at the grid cell
    # dim 4 is the binary bit representing occupancy
    force_matrix = np.zeros((GRID_DIMENSIONS, GRID_DIMENSIONS, 4))
    for i in range(GRID_DIMENSIONS):
        for j in range(GRID_DIMENSIONS):
            if grid_mass[i, j] > 0 and i >= beam_starting_x + PADDING_AMOUNT:
                force_matrix[i, j, 0] = 0
                force_matrix[i, j, 1] = gravity * grid_mass[i, j]
                force_matrix[i, j, 2] = grid_mass[i, j]
                # Binary bit at this position is a 1 when the field has a value
                force_matrix[i, j, 3] = 1
    return force_matrix


def nn_get_baseline_F_field():
    # Deformation field is the averaged vector field of deformation gradients
    deformation_field = np.zeros((GRID_DIMENSIONS, GRID_DIMENSIONS, 4))

    # Cram p2g into the system and map the deformation gradient into the field
    for p in range(n_particles):
        base = (positions[p] * inv_dx - 0.5).to_numpy().astype(int)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = np.array([i, j])
            F_entry = deformation_gradient[p].to_numpy().flatten()

            field_index = base + offset

            if (deformation_field[field_index[0], field_index[1]] != 0).all():
                # Average the deformation gradients at each position
                F_entry /= deformation_field[field_index[0], field_index[1]]
            deformation_field[field_index[0], field_index[1]] = F_entry
    return deformation_field


def sagfree_init(verify=False):
    # get lhs matrix
    particle_to_grid_static()
    row_num, col_num = compute_active_dof_rows_and_cols()
    row, col, dat = get_sparse_matrix_info_2D()
    A_sp = csc_matrix((dat, (row, col)), shape=(row_num * 2, col_num * 3))

    # Get rhs vector - basically the grid velocity/force as a vector
    b = get_active_DOF_rest_forces(row_num)

    # solve the global stage
    c = solve_least_squares(A_sp, b)

    # verify results from the global stage
    verify_global_solver(c)

    # solve the local stage
    res_array = np.zeros(col_num * 3)
    for i in range(0, col_num):
        res = find_solution(
            np.array([[c[i * 3], c[i * 3 + 2]], [c[i * 3 + 2], c[i * 3 + 1]]])
        )
        res_array[i * 3 + 0] = res[0]
        res_array[i * 3 + 1] = res[1]
        res_array[i * 3 + 2] = res[2]

    # verify results from the local stage
    verify_local_solver(res_array, col_num, c)

    # copy the results into F
    col_num = 0
    for p in range(n_particles):
        if positions[p][0] >= (beam_starting_x + 1) * dx:
            deformation_gradient[p][0, 0] = res_array[col_num * 3 + 0]
            deformation_gradient[p][1, 1] = res_array[col_num * 3 + 1]
            deformation_gradient[p][0, 1] = res_array[col_num * 3 + 2]
            deformation_gradient[p][1, 0] = res_array[col_num * 3 + 2]
            col_num += 1


def sagfree_dataset_solve_once():
    input_force_field = nn_get_baseline_force_matrix()
    sagfree_init()
    output_deformation_field = nn_get_baseline_F_field()
    return input_force_field, output_deformation_field


@app.command()
def simulate(
    solver: str = typer.Argument("solid", help="'solid' or 'fluid' (case sensitive)"),
    quiet: bool = typer.Option(False, help="Display debug logs"),
):
    global use_fluid_solver, gravity

    if solver != "solid" and solver != "liquid":
        logger.error("Solver must be 'solid' or 'liquid'")
        exit(1)

    use_fluid_solver = solver == "fluid"

    if quiet:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    initialize()
    sagfree_init()
    gui = ti.GUI("Sagfree elastic beam", res=512, background_color=0x222222)

    frame = 0
    table = Table("Key", "Description", title="Keybindings")
    table.add_row("w", "Increase simulation gravity by 10")
    table.add_row("s", "Decrease simulation gravity by 10")
    table.add_row("spacebar", "Invert simulation gravity")
    table.add_row("escape", "Exit the simulation")

    console = Console()
    console.print(table)

    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        gui.get_event()
        if gui.is_pressed("w", ti.GUI.UP):
            gravity += 10
            logger.info(f"current gravity {gravity}")
        elif gui.is_pressed("s", ti.GUI.DOWN):
            gravity -= 10
            logger.info(f"current gravity {gravity}")
        elif gui.is_pressed(" ", ti.GUI.SPACE):
            gravity = -gravity
            logger.info(f"current gravity {gravity}")

        for _ in range(25):
            clean_grid()
            particle_to_grid()
            grid_update(gravity)
            grid_to_particle()
        gui.circles(positions.to_numpy(), radius=2, color=0xED553B)
        gui.show()
        clean_grid()
        frame += 1


def check_dir(dirname: Optional[Path]):
    if not dirname:
        raise ValueError("Provided dirname is None")

    if not os.path.exists(dirname):
        raise ValueError(f"Provided dirname {dirname} does not exist")

    if not os.path.isdir(dirname):
        raise ValueError(f"Provided path {dirname} is not a directory")


@app.command()
def generate(
    samples: int = typer.Argument(100),
    input_path: Path = typer.Option(
        None, help="The path to store the neural network training inputs"
    ),
    output_path: Path = typer.Option(
        None, help="The path to store the neural network training outputs"
    ),
):
    global use_fluid_solver, gravity

    check_dir(input_path)
    check_dir(output_path)

    filename = f"ds_E={youngs_modulus}_v={poissons_ratio}_grav={gravity}_vol={point_volume}_samples={samples}"

    all_inputs = []
    all_points = []
    all_outputs = []
    for ii in track(range(samples), description="Generating datasets"):
        initialize()
        inp, outp = sagfree_dataset_solve_once()
        all_points.append(positions.to_numpy())
        all_inputs.append(inp)
        all_outputs.append(outp)

    all_inputs = np.stack(all_inputs)
    all_points = np.stack(all_points)
    all_outputs = np.stack(all_outputs)

    logger.info(f"Inputs shape: {all_inputs.shape} size: {all_inputs.nbytes * 1e-6}mb")
    logger.info(f"Outputs shape: {all_outputs.shape} size: {all_outputs.nbytes * 1e-6}mb")

    input_filename = "inp_" + filename
    output_filename = "outp_" + filename
    input_dataset_path = os.path.join(input_path, input_filename)
    output_dataset_path = os.path.join(output_path, output_filename)

    matches = list(
        filter(
            lambda x: x.startswith(input_filename),
            os.listdir(input_path),
        )
    )
    logger.debug(f"Searching for existing files in {input_path}")
    logger.debug(f"Found {len(matches)} matches")
    if len(matches) > 0:
        input_dataset_path = Path(str(input_dataset_path) + f"_{len(matches)}")
        output_dataset_path = Path(str(output_dataset_path) + f"_{len(matches)}")

    logger.info(f"Saving to inp: {input_dataset_path}, outp: {output_dataset_path}")
    np.savez_compressed(input_dataset_path, data=all_inputs, points=all_points)
    np.savez_compressed(output_dataset_path, data=all_outputs)


@app.command(help="Train the neural network", rich_help_panel="Neural Network Commands")
def train(
    input_file: Path = typer.Argument(None, help="The path to read the input data from"),
    output_file: Path = typer.Argument(
        None, help="The path to read the output data from"
    ),
):
    logger.info("Opening datasets")

    if not os.path.exists(input_file):
        raise ValueError(f"Provided path {input_file} does not exist")
    if not os.path.exists(output_file):
        raise ValueError(f"Provided path {output_file} does not exist")

    input_dataset = np.load(input_file)["data"]
    output_dataset = np.load(output_file)["data"]

    logger.info(f"Got input dataset with shape {input_dataset.shape}")
    logger.info(f"Got output dataset with shape {output_dataset.shape}")

    model = make_model(dropout=0.3)
    history = train_model(model, input_dataset, output_dataset)
    plot(history)
    save_model(model)


if __name__ == "__main__":
    app()
