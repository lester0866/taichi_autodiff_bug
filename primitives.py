import taichi as ti

SDF_PADDING = 0.1
FLOAT_EPS = 1e-5


@ti.func
def normalize(n):
    return n / length(n)


@ti.func
def length(x):
    return ti.sqrt(x.dot(x) + 1e-8)


@ti.func
def qconjugate(q, dtype):
    return q * ti.Vector([1, -1, -1, -1], dt=dtype)


@ti.func
def qmul(q, r, dtype):
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = ti.Vector([w, x, y, z], dt=dtype)
    return out / ti.sqrt(out.dot(out))  # normalize it to prevent some unknown NaN problems.


@ti.func
def qrot(q, v, dtype):
    # rot: vec4, p vec3

    return v
    # out = ti.Vector([0., v[0], v[1], v[2]], dt=dtype)
    # out = qmul(q, qmul(varr, qconjugate(q, dtype), dtype), dtype)
    # return ti.Vector([out[1], out[2], out[3]], dt=dtype)


@ti.func
def inv_trans(pos, position, rotation, dtype):
    # assert rotation.norm() > 0.9
    inv_quat = ti.Vector([rotation[0], -rotation[1], -rotation[2], -rotation[3]], dt=dtype).normalized()
    return qrot(inv_quat, pos - position, dtype)


@ti.data_oriented
class Primitive:
    def __init__(self,
                 sizes=None,
                 max_timesteps=10,
                 dtype=ti.f64):

        self.pos_dim = 3
        self.rot_dim = 4

        self.dtype = dtype
        self.max_timesteps = max_timesteps

        self.n_primitives = n_primitives = len(sizes)

        ################
        ## attributes ##
        ################

        self.softness = ti.field(dtype, shape=())

        ################################
        ## computation flow parameters #
        ################################

        self.sizes = ti.Vector.field(sizes.shape[-1], self.dtype, shape=(n_primitives))
        self.sizes.from_numpy(sizes)

        #####################
        ## storing results ##
        #####################

        self.position = ti.Vector.field(3, self.dtype, needs_grad=True, shape=(max_timesteps, n_primitives))
        self.rotation = ti.Vector.field(4, self.dtype, needs_grad=True, shape=(max_timesteps, n_primitives))

        self.position.fill(1.0)
        self.rotation.fill(1.0)

    @ti.func
    def geom_sdf(self, f, index, gird_pos):
        raise NotImplementedError

    @ti.func
    def geom_normal(self, f, index, grid_pos):
        d = ti.cast(FLOAT_EPS, self.dtype)
        n = ti.Vector.zero(self.dtype, self.pos_dim)
        for i in ti.static(range(self.pos_dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self.geom_sdf(f, index, inc) - self.geom_sdf(f, index, dec))
        return n / length(n)

    @ti.func
    def prim_sdf(self, f, index, grid_pos):
        """ Compute the signed distance function of the primitive at the given grid position. """
        grid_pos = inv_trans(grid_pos, self.position[f, index], self.rotation[f, index], self.dtype)
        return self.geom_sdf(f, index, grid_pos)

    @ti.func
    def prim_normal(self, f, index, grid_pos):
        """ Compute the normal vector of the primitive at the given grid position. """
        grid_pos = inv_trans(grid_pos, self.position[f, index], self.rotation[f, index], self.dtype)
        return qrot(self.rotation[f, index], self.geom_normal(f, index, grid_pos), self.dtype)

    @ti.func
    def prim_collider_v(self, f, index, grid_pos, dt):
        grid_pos = qrot(self.rotation[f + 1, index], grid_pos, self.dtype)
        return grid_pos

    @ti.func
    def prim_collide(self, f, index, grid_pos, v_out, dt):
        dist = self.prim_sdf(f, index, grid_pos)
        influence = min(ti.exp(-dist * self.softness[None]), 1)
        if (self.softness[None] > 0 and influence > 0.1) or dist <= 0:
            D = self.prim_normal(f, index, grid_pos)
            collider_v_at_grid = self.prim_collider_v(f, index, grid_pos, dt)

        return v_out

    @ti.func
    def collide(self, f, grid_pos, v_out, dt):
        for i in range(self.n_primitives):
            v_out = self.prim_collide(f, i, grid_pos, v_out, dt)

        return v_out


class Capsules(Primitive):
    def __init__(self, **kwargs):
        super(Capsules, self).__init__(**kwargs)
        self.r_idx, self.h_idx = 0, 1
        self.default_rot = ti.Vector.field(self.rot_dim, self.dtype, shape=())

    @ti.func
    def geom_sdf(self, f, index, grid_pos):
        p2 = self.normalize_pos(index, grid_pos)
        return length(p2) - self.sizes[index][self.r_idx]

    @ti.func
    def geom_normal(self, f, index, grid_pos):
        p2 = self.normalize_pos(index, grid_pos)
        return normalize(p2)

    @ti.func
    def normalize_pos(self, index, grid_pos):
        p2 = grid_pos
        p2[1] += self.sizes[index][self.h_idx] / 2
        p2[1] -= min(max(p2[1], 0.0), self.sizes[index][self.h_idx])
        return p2


class Boxes(Primitive):
    def __init__(self, **kwargs):
        super(Boxes, self).__init__(**kwargs)

    @ti.func
    def geom_sdf(self, f, index, grid_pos):
        # p: vec3,b: vec3
        q = ti.abs(grid_pos) - self.sizes[index]
        out = length(max(q, 0.0))
        out += min(max(q[0], max(q[1], q[2])), 0.0)
        return out
