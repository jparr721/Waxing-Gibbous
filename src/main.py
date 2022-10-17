import numpy as np
import taichi as ti
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

from praxis import *

ti.init(arch=ti.metal)

grid_dimensions = 32
beam_starting_x = 6
beam_starting_y = 13
beam_width = 20
beam_height = 6

n_particles = beam_width * beam_height * 4

dx, inv_dx = 1 / float(grid_dimensions), float(grid_dimensions)
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
position = ti.Vector.field(2, dtype=float, shape=n_particles)
velocity = ti.Vector.field(2, dtype=float, shape=n_particles)
affine_velocity = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
deformation_gradient = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)
color = ti.field(dtype=int, shape=n_particles)
plastic_deformation = ti.field(dtype=float, shape=n_particles)
grid_velocity = ti.Vector.field(2, dtype=float, shape=(grid_dimensions, grid_dimensions))
grid_mass = ti.field(dtype=float, shape=(grid_dimensions, grid_dimensions))
grid_index = ti.field(dtype=int, shape=(grid_dimensions, grid_dimensions))


@ti.kernel
def clean_grid():
    for i, j in grid_mass:
        grid_velocity[i, j] = [0, 0]
        grid_mass[i, j] = 0


@ti.kernel
def particle_to_grid():
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
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
        if i < 3 and grid_velocity[i, j][0] < 0:
            grid_velocity[i, j][0] = 0
        if i > grid_dimensions - 3 and grid_velocity[i, j][0] > 0:
            grid_velocity[i, j][0] = 0
        if j < 3 and grid_velocity[i, j][1] < 0:
            grid_velocity[i, j][1] = 0
        if j > grid_dimensions - 3 and grid_velocity[i, j][1] > 0:
            grid_velocity[i, j][1] = 0

        # fixed left side of the beam
        if i < beam_starting_x + 3:
            grid_velocity[i, j] = [0, 0]


@ti.kernel
def grid_to_particle():
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_velocity[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        velocity[p], affine_velocity[p] = new_v, new_C
        position[p] += dt * velocity[p]


@ti.kernel
def initialize():
    for i in range(beam_width * 2):
        for j in range(beam_height * 2):
            p_count = j + i * beam_height * 2
            position[p_count] = ti.Matrix(
                [
                    (float(beam_starting_x) + i * 0.5 + 0.25) * dx,
                    (float(beam_starting_y) + j * 0.5 + 0.25) * dx,
                ]
            )
            color[p_count] = 1
            velocity[p_count] = ti.Matrix([0, 0])
            deformation_gradient[p_count] = ti.Matrix([[1, 0], [0, 1]])
            plastic_deformation[p_count] = 1


@ti.kernel
def particle_to_grid_static():
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        R, S = ti.polar_decompose(deformation_gradient[p])
        cauchy = (
            2 * mu * (deformation_gradient[p] - R)
            + lambda_
            * (
                R.transpose() @ deformation_gradient[p] - ti.Matrix.identity(ti.f32, 2)
            ).trace()
            * R
        ) @ deformation_gradient[p].transpose()
        affine = (-point_volume * 4 * inv_dx * inv_dx) * cauchy
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_velocity[base + offset] += weight * (affine @ dpos)
            grid_mass[base + offset] += weight * p_mass


def compute_active_dof_rows_and_cols():
    for i in range(0, grid_dimensions):
        for j in range(0, grid_dimensions):
            grid_index[i, j] = -1

    rn = 0
    for i in range(0, grid_dimensions):
        for j in range(0, grid_dimensions):
            if grid_mass[i, j] > 0 and i >= beam_starting_x + 3:
                grid_index[i, j] = rn
                rn = rn + 1

    cn = 0
    for i in range(0, n_particles):
        if position[i][0] > (beam_starting_x + 1) * dx:  # extra columns for particles
            cn = cn + 1

    return rn, cn


def get_sparse_matrix_info_2D():
    arr_len = 0
    for p in range(0, n_particles):
        if position[p][0] > (beam_starting_x + 1) * dx:
            basex = int(position[p][0] * inv_dx - 0.5)
            basey = int(position[p][1] * inv_dx - 0.5)
            fx = np.array(
                [
                    float(position[p][0] * inv_dx - basex),
                    float(position[p][1] * inv_dx - basey),
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
                        and basex + i >= beam_starting_x + 3
                    ):
                        arr_len = arr_len + 1

    row = np.zeros(arr_len * 4)
    col = np.zeros(arr_len * 4)
    dat = np.zeros(arr_len * 4)
    arr_len = 0
    col_num = 0
    for p in range(0, n_particles):
        if position[p][0] > (beam_starting_x + 1) * dx:
            basex = int(position[p][0] * inv_dx - 0.5)
            basey = int(position[p][1] * inv_dx - 0.5)
            fx = np.array(
                [
                    float(position[p][0] * inv_dx - basex),
                    float(position[p][1] * inv_dx - basey),
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
                        and basex + i >= beam_starting_x + 3
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
                        arr_len = arr_len + 1
            col_num = col_num + 1
    return row, col, dat


def get_active_DOF_rest_forces(row_num):
    rhs = np.zeros(row_num * 2)
    row_num = 0
    for i in range(0, grid_dimensions):
        for j in range(0, grid_dimensions):
            if grid_mass[i, j] > 0 and i >= beam_starting_x + 3:
                rhs[row_num * 2 + 0] = 0
                rhs[row_num * 2 + 1] = gravity * grid_mass[i, j]
                row_num = row_num + 1
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
    for p in position:
        base = (position[p] * inv_dx - 0.5).cast(int)
        fx = position[p] * inv_dx - base.cast(float)
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
        if position[p][0] >= (beam_starting_x + 1) * dx:
            deformation_gradient[p][0, 0] = c[col_num * 3 + 0]
            deformation_gradient[p][1, 1] = c[col_num * 3 + 1]
            deformation_gradient[p][0, 1] = c[col_num * 3 + 2]
            deformation_gradient[p][1, 0] = c[col_num * 3 + 2]
            col_num = col_num + 1

    test_particle_to_grid_static()

    for i in range(0, grid_dimensions):
        for j in range(0, grid_dimensions):
            if grid_mass[i, j] > 0 and i >= beam_starting_x + 3:
                if not np.isclose(grid_velocity[i, j][0], 0, atol=1e-6):
                    print("Error in global step: ", i, j, grid_velocity[i, j][0] - 0)
                if not np.isclose(
                    grid_velocity[i, j][1], gravity * grid_mass[i, j], atol=1e-6
                ):
                    print(
                        "Error in global step: ",
                        i,
                        j,
                        grid_velocity[i, j][1] - gravity * grid_mass[i, j],
                    )


def verify_local_solver(res_array, col_num, c):
    for i in range(0, col_num):
        affine = evaluate_affine_mapping(
            res_array[i * 3 + 0], res_array[i * 3 + 1], res_array[i * 3 + 2]
        )
        res_vec = np.array([affine[0, 0], affine[1, 1], affine[1, 0]])
        tar_vec = np.array([c[i * 3], c[i * 3 + 1], c[i * 3 + 2]])
        if not np.allclose(res_vec, tar_vec, atol=1e-6):
            print(
                "Error in local step: ",
                c[i * 3],
                c[i * 3 + 1],
                c[i * 3 + 2],
                res_vec,
                tar_vec,
                res_vec - tar_vec,
            )


####################################################################
# start from here
####################################################################

initialize()


def sagfree_init():
    # get lhs matrix
    particle_to_grid_static()
    row_num, col_num = compute_active_dof_rows_and_cols()
    row, col, dat = get_sparse_matrix_info_2D()
    A_sp = csc_matrix((dat, (row, col)), shape=(row_num * 2, col_num * 3))

    # get rhs vector
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
    for p in range(0, n_particles):
        if position[p][0] >= (beam_starting_x + 1) * dx:
            deformation_gradient[p][0, 0] = res_array[col_num * 3 + 0]
            deformation_gradient[p][1, 1] = res_array[col_num * 3 + 1]
            deformation_gradient[p][0, 1] = res_array[col_num * 3 + 2]
            deformation_gradient[p][1, 0] = res_array[col_num * 3 + 2]
            col_num = col_num + 1


sagfree_init()
gui = ti.GUI("Sagfree elastic beam", res=512, background_color=0x222222)

frame = 0
while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):

    if frame > 200:
        gravity = -20.0 * np.sin(frame)
    frame = frame + 1

    for s in range(int(2e-3 // dt)):
        clean_grid()
        particle_to_grid()
        grid_update(gravity)
        grid_to_particle()
    gui.circles(position.to_numpy(), radius=1.5, color=0xED553B)

    gui.show()

    clean_grid()
