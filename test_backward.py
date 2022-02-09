import numpy as np
import taichi as ti

from mpm import MPMSimulator
from primitives import Boxes, Capsules

ti.init(ti.cuda, device_memory_fraction=0.9, ad_stack_size=64)

primitives = []
primitives.append(Boxes(sizes=np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])))
primitives.append(Capsules(sizes=np.array([[0.5, 0.5]])))
primitives = tuple(primitives)

sim = MPMSimulator(primitives)

loss = ti.field(ti.f64, shape=(), needs_grad=True)
with ti.Tape(loss=loss):
    sim.grid_op(0)
