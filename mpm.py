import taichi as ti

dim = 3
n_grid = 64
quality = 1
dtype = ti.f64


@ti.data_oriented
class MPMSimulator:
    def __init__(self, primitives=()):
        self.primitives = primitives
        self.n_primitive = len(primitives)

        init_scalar = lambda shape: ti.field(dtype=dtype, shape=shape)
        self.dx, self.dt = init_scalar(()), init_scalar(())

        self.dx[None] = (1 / n_grid)
        self.dt[None] = 0.5e-4 / quality
        self.res = res = (n_grid, n_grid, n_grid)
        self.grid_v_in = ti.Vector.field(dim, dtype=dtype, shape=res, needs_grad=True)  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=dtype, shape=res, needs_grad=True)  # grid node mass

    @ti.kernel
    def grid_op(self, f: ti.i32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-12:  # No need for epsilon here, 1e-10 is to prevent potential numerical problems ..
                v_out = (1 / self.grid_m[I]) * self.grid_v_in[I]  # Momentum to velocity

                if ti.static(self.n_primitive > 0):
                    for i in ti.static(range(self.n_primitive)):
                        v_out = self.primitives[i].collide(f, I * self.dx[None], v_out, self.dt[None])
