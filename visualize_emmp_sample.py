import numpy as np
from models import load_pretrained
import loader 
import os
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
from copy import deepcopy
from loader import get_dataloader
from loader.Pouring_dataset import Pouring
from loader.latent_dataset import LatentDataset
from models.Latent_sampler import get_optimized_GMM_latent
import pickle
from utils import LieGroup_torch as lie
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from vis_utils.open3d_utils import (
    get_mesh_bottle, 
    get_mesh_mug, 
    visualize_pouring_traj
)
import platform
import glob
import threading
import time

from utils.utils import SE3smoothing

class AppWindow:

    def __init__(self):

        # argparser
        parser = argparse.ArgumentParser()
        parser.add_argument("--pretrained_root", "-p", default='pretrained/Pouring/EMMP/AE')
        parser.add_argument("--identifier", "-i", default="noreg")
        parser.add_argument("--config_file", "-co", default="EMMP_ae_noreg.yml")
        parser.add_argument("--ckpt_file", "-ck", default="model_best.pkl")
        parser.add_argument("--device", type=str, default='any')
        args, unknown = parser.parse_known_args()

        # Setup device
        if args.device == "cpu":
            self.device = f"cpu"
        elif args.device == "any":
            self.device = f"cuda"
        else:
            self.device = f"cuda:{args.device}"

        self.model, cfg = load_pretrained(
            args.identifier,
            args.config_file,
            args.ckpt_file,
            root=args.pretrained_root
        )
        self.model.to(self.device)
        # Setup Dataloader
        data_dict = {
            'dataset': 'Pouring',
            'root': 'datasets/',
            'batch_size': 75,
            'n_workers': 4,
            'split': 'training',
            'skip_size': 1,
            'shuffle': False,
            'augmentation': True,
            }
        self.dataloader = get_dataloader(data_dict)
        self.gmm, _ = get_optimized_GMM_latent(self.model, self.dataloader)
        sample_size_big = 10000
        # below part is added since the GMM samples are same everytime.
        randidx = torch.randperm(sample_size_big)
        self.z_sample_all = self.gmm.sample(sample_size_big)[0][randidx]
        self.current_sample_idx = 0
        
        # Variables initialization
        self.target_fps = 60
        self.dt_video = 1 / self.target_fps
        self.skip_size = 5
        self.rot_bottle = torch.tensor(0.0)
        # self.tau = torch.tensor([0.0, 0.0, 0.6, 0.0, 0.2, 1.0, 0.0]).to(self.device)
        self.tau = torch.tensor([0.0, 0.0, 0.6, 0.0, 0.2, 0.0]).to(self.device)
        self.traj = None
        self.mug_T = torch.eye(4)
        self.video_vis_mode = False
        
        
        # Thread initialization
        self.event = threading.Event()
        self.thread_video = threading.Thread(target=self.update_trajectory_video, daemon=True)

        # parameters 
        image_size = [1680, 960]
        
        # mesh table
        self.mesh_box = o3d.geometry.TriangleMesh.create_box(
            width=2, height=2, depth=0.03
        )
        self.mesh_box.translate([-1, -1, -0.03])
        self.mesh_box.paint_uniform_color([222/255,184/255,135/255])
        self.mesh_box.compute_vertex_normals()

        # frame
        self.frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1
        )

        # object material
        self.mat = rendering.MaterialRecord()
        self.mat.shader = 'defaultLit'
        self.mat.base_color = [1.0, 1.0, 1.0, 0.9]
        mat_prev = rendering.MaterialRecord()
        mat_prev.shader = 'defaultLitTransparency'
        mat_prev.base_color = [1.0, 1.0, 1.0, 0.7]
        mat_coord = rendering.MaterialRecord()
        mat_coord.shader = 'defaultLitTransparency'
        mat_coord.base_color = [1.0, 1.0, 1.0, 0.87]

        ######################################################
        ################# STARTS FROM HERE ###################
        ######################################################

        # set window
        self.window = gui.Application.instance.create_window(str(datetime.now().strftime('%H%M%S')), width=image_size[0], height=image_size[1])
        w = self.window
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        # camera viewpoint
        self._scene.scene.camera.look_at(
            [0, 0, 0], # camera lookat
            [0.7, 0, 0.9], # camera position
            [0, 0, 1] # fixed
        )

        # other settings
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.3, 0.3, -0.9))
        self._scene.scene.set_background([1.0, 1.0, 1.0, 1.0], image=None)

        ############################################################
        ######################### MENU BAR #########################
        ############################################################
        
        # menu bar initialize
        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        # initialize collapsable vert
        inference_config = gui.CollapsableVert("Inference config", 0.25 * em,
                                        gui.Margins(em, 0, 0, 0))
        
        # sample
        self._sample_button = gui.Button("Sample!")
        self._sample_button.horizontal_padding_em = 0.5
        self._sample_button.vertical_padding_em = 0
        self._sample_button.set_on_clicked(self._set_sampler)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._sample_button)
        h.add_stretch()

        # add
        inference_config.add_child(gui.Label("Sample bottle trajectory"))
        inference_config.add_child(h)
        
        # Visualize type
        self._video_button = gui.Button("Video")
        self._video_button.horizontal_padding_em = 0.5
        self._video_button.vertical_padding_em = 0
        self._video_button.set_on_clicked(self._set_vis_mode_video)
        self._afterimage_button = gui.Button("Afterimage")
        self._afterimage_button.horizontal_padding_em = 0.5
        self._afterimage_button.vertical_padding_em = 0
        self._afterimage_button.set_on_clicked(self._set_vis_mode_afterimage)
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._video_button)
        h.add_child(self._afterimage_button)
        h.add_stretch()

        # add
        inference_config.add_child(gui.Label("Visualize type"))
        inference_config.add_child(h)

        # skip size
        self._skip_size_silder = gui.Slider(gui.Slider.INT)
        self._skip_size_silder.set_limits(1, 100)
        self._skip_size_silder.int_value = self.skip_size
        self._skip_size_silder.set_on_value_changed(self._set_skip_size)
        
        # add
        inference_config.add_fixed(separation_height)
        inference_config.add_child(gui.Label("Skip size"))
        inference_config.add_child(self._skip_size_silder)


        # cup_x
        self._cup_x_editor = gui.TextEdit()
        self._cup_x_editor.text_value = str(int(self.tau[0] * 100))
        self._cup_x_editor.set_on_value_changed(self._set_cup_x)
        
        # cup_y
        self._cup_y_editor = gui.TextEdit()
        self._cup_y_editor.text_value = str(int(self.tau[1] * 100))
        self._cup_y_editor.set_on_value_changed(self._set_cup_y)

        # add cup x, y
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._cup_x_editor)
        h.add_child(self._cup_y_editor)
        h.add_stretch()
        # add
        inference_config.add_child(gui.Label("cup position in cm (x, y)"))
        inference_config.add_child(h)

        # bottle_x
        self._bottle_x_editor = gui.TextEdit()
        self._bottle_x_editor.text_value = str(int(self.tau[2] * 100))
        self._bottle_x_editor.set_on_value_changed(self._set_bottle_x)
        
        # bottle_y
        self._bottle_y_editor = gui.TextEdit()
        self._bottle_y_editor.text_value = str(int(self.tau[3] * 100))
        self._bottle_y_editor.set_on_value_changed(self._set_bottle_y)
        
        # add bottle x, y
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._bottle_x_editor)
        h.add_child(self._bottle_y_editor)
        h.add_stretch()
        # add
        inference_config.add_child(gui.Label("bottle position in cm (x, y)"))
        inference_config.add_child(h)


        # water_weight
        self._water_weight_editor = gui.TextEdit()
        self._water_weight_editor.text_value = str(int(self.tau[4] * 1_000))
        self._water_weight_editor.set_on_value_changed(self._set_water_weight)
        
        # bottle_z_rot
        self._bottle_rot_editor = gui.TextEdit()
        self._bottle_rot_editor.text_value = str(int(self.rot_bottle * 180 / np.pi))
        self._bottle_rot_editor.set_on_value_changed(self._set_bottle_rot)
        
        # add bottle x, y
        h = gui.Horiz(0.25 * em)  # row 1
        h.add_stretch()
        h.add_child(self._water_weight_editor)
        h.add_child(self._bottle_rot_editor)
        h.add_stretch()
        # add
        inference_config.add_child(gui.Label("WW(200~410), BR in degrees"))
        inference_config.add_child(h)

        # add 
        self._settings_panel.add_child(inference_config)

        # add scene
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # initial scene
        self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)
    
        # mesh list
        self.bottle_idx = 1
        self.mug_idx = 1
        
        # load bottle
        self.mesh_bottle = get_mesh_bottle(
            root='./3dmodels/bottles', 
            bottle_idx=self.bottle_idx
        )
        self.mesh_bottle.translate([0, 0, -0.145]) # grasp point height = 14.5 cm
        self.mesh_bottle.compute_vertex_normals()

        # color template (2023 pantone top 10 colors)
        rgb = np.zeros((10, 3))
        rgb[0, :] = [208, 28, 31] # fiery red
        rgb[1, :] = [207, 45, 113] # beetroot purple
        rgb[2, :] = [249, 77, 0] # tangelo
        rgb[3, :] = [250, 154, 133] # peach pink
        rgb[4, :] = [247, 208, 0] # empire yellow
        rgb[5, :] = [253, 195, 198] # crystal rose
        rgb[6, :] = [57, 168, 69] # classic green
        rgb[7, :] = [193, 219, 60] # love bird 
        rgb[8, :] = [75, 129, 191] # blue perennial 
        rgb[9, :] = [161, 195, 218] # summer song
        rgb = rgb / 255  

        # new bottle coloring
        bottle_vertices = np.asarray(self.mesh_bottle.vertices) 
        bottle_normals = np.asarray(self.mesh_bottle.vertex_normals)
        bottle_colors = np.ones_like(bottle_normals)
        bottle_colors[:, :3] = rgb[8]
        # z_values = bottle_vertices[:, 2]
        # print(np.max(z_values), np.min(z_values))
        # bottle_colors[np.logical_and(z_values > 0.17, z_values < 0.2)] = rgb[6]
        self.mesh_bottle.vertex_colors = o3d.utility.Vector3dVector(
            bottle_colors
        )

        # load mug
        self.mesh_mug = get_mesh_mug(
            root='./3dmodels/mugs', 
            mug_idx=self.mug_idx
        )
        self.mesh_mug.paint_uniform_color(rgb[2] * 0.6)
        self.mesh_mug.compute_vertex_normals()

    
    def end_thread(self, thread:threading.Thread):
        self.event.set()
        while thread.is_alive():
            time.sleep(0.1)
        self.event.clear()
    
    def reset_threads(self):
        if self.thread_video.is_alive():
            self.end_thread(self.thread_video)
        self.thread_video = threading.Thread(
                            target=self.update_trajectory_video,
                            daemon=True)
    
    
    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
                                            height)

    def _update_traj(self):
        self.remove_trajectory()
        self.update_trajectory()
        
    def _set_vis_mode_video(self):
        self.video_vis_mode = True
        self.reset_threads()
        self.thread_video.start()

    def _set_vis_mode_afterimage(self):
        self.video_vis_mode = False
        self.reset_threads()
        self.remove_trajectory()
        self.update_trajectory()

    def _set_sampler(self):
        self.reset_threads()
        self.remove_trajectory()
        self.sample_trajectories()
        if self.video_vis_mode:
            self._set_vis_mode_video()
        else:
            self.update_trajectory()

    def _set_skip_size(self, value):
        self.skip_size = int(value)
        
    def _set_cup_x(self, value):
        self.tau[0] = float(value) / 100
        self.mug_T[0, 3] = self.tau[0]
        self.sample_trajectories()

    def _set_cup_y(self, value):
        self.tau[1] = float(value) / 100
        self.mug_T[1, 3] = self.tau[1]
        self.sample_trajectories()
        
    def _set_bottle_x(self, value):
        self.tau[2] = float(value) / 100
        self.sample_trajectories()

    def _set_bottle_y(self, value):
        self.tau[3] = float(value) / 100
        self.sample_trajectories()
        
    def _set_water_weight(self, value):
        self.tau[4] = float(value) / 1000
        self.sample_trajectories()
    
    def _set_bottle_rot(self, value):
        self.rot_bottle = torch.tensor(float(value) / 180 * np.pi)
        # self.tau[5] = torch.cos(self.rot_bottle)
        # self.tau[6] = torch.sin(self.rot_bottle)
        self.tau[5] = self.rot_bottle
        self.sample_trajectories()
        

    def sample_trajectories(self):
        z_gen = self.z_sample_all[self.current_sample_idx]
        z_gen = torch.from_numpy(z_gen).to(torch.float).to(self.device).unsqueeze(0)
        tau = self.tau.unsqueeze(0)
        x_gen = self.model.decode(z_gen, tau).detach().cpu()
        T_gen = x_gen[0].reshape(-1, 3, 4)
        zero_one = torch.tensor([0, 0, 0, 1]).unsqueeze(0).unsqueeze(0).repeat(len(T_gen), 1, 1)
        T_gen = torch.cat([T_gen, zero_one], dim=-2)
        self.traj = T_gen
        # smoothing
        self.traj = SE3smoothing(self.traj.unsqueeze(0), mode='savgol').squeeze()
        self.current_sample_idx += 1


    def update_trajectory(self):
        # load data
        if self.traj is None:
            self.sample_trajectories()
            
        mesh_mug_ = deepcopy(self.mesh_mug)
        frame_ = deepcopy(self.frame)
        mesh_mug_.transform(self.mug_T)
        frame_.transform(self.mug_T)
        # update initials
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame_init', frame_, self.mat)
        self._scene.scene.add_geometry('mug_init', mesh_mug_, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)

        # update trajectory
        for idx in range(0, len(self.traj), self.skip_size):
            mesh_bottle_ = deepcopy(self.mesh_bottle)
            frame_bottle = deepcopy(self.frame)
            T = self.traj[idx]
            mesh_bottle_.transform(T)
            frame_bottle.transform(T)

            self._scene.scene.add_geometry(f'bottle_{idx}', mesh_bottle_, self.mat)
            if idx == 0:
                self._scene.scene.add_geometry(f'coord_{idx}', frame_bottle, self.mat)

    def update_trajectory_video(self):
                
        # load data
        if self.traj is None:
            self.sample_trajectories()
        
        # update initials
        self.mesh_mug_ = deepcopy(self.mesh_mug)
        self.frame_ = deepcopy(self.frame)
        self.mesh_mug_.transform(self.mug_T)
        self.frame_.transform(self.mug_T)
        self.table_frame = deepcopy(self.frame)
        # t_last_idx = [0] * len(range(0, len(self.traj), self.skip_size))

        # update trajectory
        for idx in range(0, len(self.traj), self.skip_size):
            if self.event.is_set():
                break
            self.idx = idx
            self.mesh_bottle_ = deepcopy(self.mesh_bottle)
            self.frame_bottle = deepcopy(self.frame)
            T = self.traj[idx]
            self.mesh_bottle_.transform(T)
            self.frame_bottle.transform(T)
            # Update geometry
            gui.Application.instance.post_to_main_thread(self.window, self.update_scene)
            time.sleep(0.05)

    def update_scene(self):
        
        self._scene.scene.clear_geometry()
        self._scene.scene.add_geometry('frame_init', self.frame_, self.mat)
        self._scene.scene.add_geometry('mug_init', self.mesh_mug_, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)
        self._scene.scene.add_geometry(
            f'bottle_{self.idx}', self.mesh_bottle_, self.mat
        )
        self._scene.scene.add_geometry(
            f'coord_{self.idx}', self.frame_bottle, self.mat)

    def remove_trajectory(self):
        self._scene.scene.clear_geometry()
        # self._scene.scene.add_geometry('frame_init', self.frame, self.mat)
        self._scene.scene.add_geometry('box_init', self.mesh_box, self.mat)

if __name__ == "__main__":

    gui.Application.instance.initialize()

    w = AppWindow()

    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()