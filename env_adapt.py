import sys
sys.path.insert(0, "composite")

from isaacgym import gymapi
from env import parse_kwarg
from env import ICCGANHumanoidTarget
from env import heading_zup, DiscriminatorConfig

import numpy as np
import torch


class ICCGANHumanoidLowFriction(ICCGANHumanoidTarget):

    def __init__(self, *args, **kwargs):
        self.ground_friction = parse_kwarg(kwargs, "ground_friction", 1.)
        super().__init__(*args, **kwargs)

    def add_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(*self.vector_up(1.0))
        plane_params.static_friction = self.ground_friction
        plane_params.dynamic_friction = self.ground_friction
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def add_actor(self, env, i):
        if self.ground_friction != 1:
            left_foot = self.gym.find_actor_rigid_body_handle(env, 0, "left_foot")
            right_foot = self.gym.find_actor_rigid_body_handle(env, 0, "right_foot")
            rb_shape = self.gym.get_actor_rigid_body_shape_indices(env, 0)
            rb_shape_props = self.gym.get_actor_rigid_shape_properties(env, 0)
            rb_shape_props[rb_shape[left_foot].start].friction = self.ground_friction
            rb_shape_props[rb_shape[left_foot].start].rolling_friction = self.ground_friction
            rb_shape_props[rb_shape[left_foot].start].torsion_friction = self.ground_friction
            rb_shape_props[rb_shape[right_foot].start].friction = self.ground_friction
            rb_shape_props[rb_shape[right_foot].start].rolling_friction = self.ground_friction
            rb_shape_props[rb_shape[right_foot].start].torsion_friction = self.ground_friction
            self.gym.set_actor_rigid_shape_properties(env, 0, rb_shape_props)

class ICCGANHumanoidTerrain(ICCGANHumanoidTarget):

    def add_ground(self):
        rand_state = np.random.get_state()

        from isaacgym.terrain_utils import SubTerrain, convert_heightfield_to_trimesh, \
            pyramid_sloped_terrain, random_uniform_terrain
        
        vertical_scale = 0.005
        self.horizontal_scale = 0.1
        slope_threshold = 0.5
        friction = 1.0
        border = 10

        n_envs = len(self.envs)
        env_spacing = 3
        sub_terrain_size_x = env_spacing+env_spacing
        sub_terrain_size_y = env_spacing+env_spacing
        sub_grids_y = int(sub_terrain_size_y / self.horizontal_scale)
        sub_grids_x = int(sub_terrain_size_x / self.horizontal_scale)

        n_envs_per_row = int(n_envs**0.5)
        field_y = sub_terrain_size_y*n_envs_per_row + border*2*sub_terrain_size_y
        field_x = sub_terrain_size_x*int(np.ceil(n_envs/n_envs_per_row)) + border*2*sub_terrain_size_x

        grids_y = int(field_y / sub_terrain_size_y)
        grids_x = int(field_x / sub_terrain_size_x)

        height_map_raw = np.zeros((grids_x*sub_grids_x, grids_y*sub_grids_y), dtype=np.int16)
        for i in range(grids_x):
            for j in range(grids_y):
                x0, x1 = i*sub_grids_x, (i+1)*sub_grids_x
                y0, y1 = j*sub_grids_y, (j+1)*sub_grids_y
                terrain = SubTerrain(width=sub_grids_x, length=sub_grids_y, vertical_scale=vertical_scale, horizontal_scale=self.horizontal_scale)

                pyramid_sloped_terrain(terrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3]))
                random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.05, downsampled_scale=0.2)
                height_map_raw[x0:x1, y0:y1] = terrain.height_field_raw
        
        if self.viewer is None:
            heightfield_raw = height_map_raw
        else:
            heightfield_raw = height_map_raw[:21*sub_grids_x, :21*sub_grids_y]
        self.height_map_raw = height_map_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield_raw, self.horizontal_scale, vertical_scale, slope_threshold)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -border*sub_terrain_size_x-env_spacing
        tm_params.transform.p.y = -border*sub_terrain_size_y-env_spacing
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = friction
        tm_params.dynamic_friction = friction
        tm_params.restitution = 0
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(order='C'), triangles.flatten(order='C'), tm_params)
        self.height_map = torch.tensor(heightfield_raw*vertical_scale-tm_params.transform.p.z, dtype=torch.float32).to(self.device)
        
        env_origins = []
        for env in self.envs:
            p = self.gym.get_env_origin(env)
            env_origins.append([p.x, p.y])
        env_origins = torch.tensor(env_origins).to(self.device)
        map_offset = torch.tensor([tm_params.transform.p.x, tm_params.transform.p.y]).to(self.device)
        self.env_origins = env_origins.sub_(map_offset)
        self.terrain_border = 1, \
            torch.tensor([
                self.height_map.shape[0]*self.horizontal_scale-1,
                self.height_map.shape[1]*self.horizontal_scale-1],
            dtype=torch.float32, device=self.device)

        x_range, y_range = (-10, 25), (-17, 18)
        self.height_sampling_points = torch.tensor(np.reshape(
            np.multiply(self.horizontal_scale, [(i, j) for i in range(*x_range) for j in range(*y_range)]),
            (x_range[1]-x_range[0], y_range[1]-y_range[0], 2)
        ), dtype=torch.float32, device=self.device)
        self.height_map_scale = 1/1.5
        self.height_map_offset = 0

        np.random.set_state(rand_state)

    def ground_height(self, p, env_ids=None):
        if env_ids is None:
            p = p[..., :2] + self.env_origins
        else:
            p = p[..., :2] + self.env_origins[env_ids]
        p = (p/self.horizontal_scale).long()
        x = torch.clip(p[..., 0], 0, self.height_map.shape[0]-1)
        y = torch.clip(p[..., 1], 0, self.height_map.shape[1]-1)
        return self.height_map[x, y]

    def normalize_height_map(self, height_map):
        height_map = height_map - height_map[:, 0, 17, 24].view(-1, 1, 1, 1)
        if self.height_map_scale:
            height_map = height_map.mul_(self.height_map_scale)
        if self.height_map_offset:
            height_map.add_(self.height_map_offset)
        return height_map
    
    def observe_ground_height(self, env_ids=None):
        if env_ids is None:
            height_map = observe_height_map(self.char_root_tensor, self.height_sampling_points, self.env_origins, self.horizontal_scale, self.height_map)
        else:
            height_map = observe_height_map(self.char_root_tensor[env_ids], self.height_sampling_points, self.env_origins[env_ids], self.horizontal_scale, self.height_map)
        height_map = height_map.view(-1, 1, *self.height_sampling_points.shape[:-1])
        height_map = self.normalize_height_map(height_map)
        return height_map

    def _observe(self, env_ids):
        if env_ids is None:
            self.info["map"] = self.observe_ground_height()
        else:
            self.info["map"][env_ids] = self.observe_ground_height(env_ids)
        return super()._observe(env_ids)

@torch.jit.script
def observe_height_map(root_state: torch.Tensor, sampling_points: torch.Tensor, env_origins: torch.Tensor, horizontal_scale: float, height_map: torch.Tensor):
    root_orient = root_state[:, 3:7]
    x = sampling_points[..., 0].view(-1)
    # if UP_AXIS == 2:
    y = sampling_points[..., 1].view(-1)
    root_pos = root_state[:, :2]
    heading = heading_zup(root_orient).unsqueeze_(-1)
    c = torch.cos(heading)
    s = torch.sin(heading)
    x, y = c*x-s*y, s*x+c*y                         # N_envs x N_points
    p = torch.stack((x, y), -1)                     # N_envs x N_points x 2
    p = p + (root_pos + env_origins).unsqueeze_(1)  # N_envs x N_points x 2
    p = (p/horizontal_scale).long().view(-1, 2)     # (N_envs x N_points)
    x = torch.clip(p[:, 0], 0, height_map.shape[0]-1)
    y = torch.clip(p[:, 1], 0, height_map.shape[1]-1)
    h = height_map[x, y]
    return h
