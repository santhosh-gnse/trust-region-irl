#!/usr/bin/env python3
"""
mujoco_xml_viewer.py

Simple MuJoCo + GLFW viewer that opens an MJCF (XML) file, runs the simulation,
and displays it in a window.

Usage:
    python mujoco_xml_viewer.py path/to/model.xml

Controls:
    SPACE - pause / unpause simulation
    H     - hide / show overlay
    TAB   - toggle camera mode ("static" / "follow")
    S     - slow down (divide run speed by 2)
    F     - speed up (multiply run speed by 2)
    Mouse - rotate / pan / zoom (left/right/middle)
    ESC   - close window
"""

import sys
import time
import argparse
from itertools import cycle
import numpy as np

import glfw
import mujoco


class MujocoViewer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.dt = float(model.opt.timestep)  # simulation timestep

        # input state
        self.button_left = self.button_right = self.button_middle = False
        self.last_x = self.last_y = 0

        # timing & simulation speed
        self.run_speed_factor = 1.0
        self.paused = False
        self.hide_menu = False

        # overlay / UI
        self.overlay = {}
        self.font_scale = 100  # choose a default font scale

        # glfw + window
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)

        primary_monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(primary_monitor)
        window_width, window_height = video_mode.size

        self.window = glfw.create_window(width=window_width, height=window_height, title="MuJoCo XML Viewer", monitor=None, share=None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(self.window)

        # callbacks
        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_scroll_callback(self.window, self.scroll)

        # MuJoCo rendering objects
        self.scene = mujoco.MjvScene(self.model, 1000)
        self.scene_option = mujoco.MjvOption()

        # camera
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.all_camera_modes = ("static", "follow")
        self.camera_mode_iter = cycle(self.all_camera_modes)
        self.camera_mode = next(self.camera_mode_iter)
        self.camera_mode_target = self.camera_mode
        self.set_camera()

        # viewport & context
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale(self.font_scale))

        # timing bookkeeping
        self._last_time = time.time()

    # ---------- input callbacks ----------
    def mouse_button(self, window, button, act, mods):
        self.button_left = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        self.last_x, self.last_y = glfw.get_cursor_pos(self.window)

    def mouse_move(self, window, x_pos, y_pos):
        if not (self.button_left or self.button_right or self.button_middle):
            return

        dx = x_pos - self.last_x
        dy = y_pos - self.last_y
        self.last_x, self.last_y = x_pos, y_pos

        width, height = glfw.get_window_size(self.window)

        mod_shift = glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(self.window,
                                                                                                  glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(self.model, action, dx / width, dy / height, self.scene, self.camera)

    def keyboard(self, window, key, scancode, act, mods):
        # only trigger on key release so toggles are stable
        if act != glfw.RELEASE:
            return
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_H:
            self.hide_menu = not self.hide_menu
        elif key == glfw.KEY_TAB:
            self.camera_mode_target = next(self.camera_mode_iter)
        elif key == glfw.KEY_S:
            self.run_speed_factor = max(1e-6, self.run_speed_factor / 2.0)
        elif key == glfw.KEY_F:
            self.run_speed_factor *= 2.0
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.window, True)

    def scroll(self, window, x_offset, y_offset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y_offset, self.scene, self.camera)

    # ---------- rendering / simulation ----------
    def create_overlay(self, fps):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        self.overlay.clear()
        self.overlay[bottomright] = ["Framerate:", f"{int(fps)}"]
        self.overlay[topleft] = ["", ""]
        self.overlay[topleft][0] += "Press SPACE to pause.\n"
        self.overlay[topleft][1] += "\n"
        self.overlay[topleft][0] += "Press H to hide the menu.\n"
        self.overlay[topleft][1] += "\n"
        self.overlay[topleft][0] += "Press TAB to switch cameras.\n"
        self.overlay[topleft][1] += "\n"
        self.overlay[topleft][0] += "Camera mode:\n"
        self.overlay[topleft][1] += self.camera_mode + "\n"
        self.overlay[topleft][0] += f"Run speed = {self.run_speed_factor:.3f} x real time"
        self.overlay[topleft][1] += "[S]lower, [F]aster"

    def set_camera(self):
        if self.camera_mode_target == "static" and self.camera_mode != "static":
            self.camera.fixedcamid = 0
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.trackbodyid = -1
            self.camera.distance = 15.0
            self.camera.elevation = -45.0
            self.camera.azimuth = 90.0

        if self.camera_mode_target == "follow" and self.camera_mode != "follow":
            self.camera.fixedcamid = -1
            self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.camera.trackbodyid = 0
            self.camera.distance = 3.5
            self.camera.elevation = 0.0
            self.camera.azimuth = 90.0

        self.camera_mode = self.camera_mode_target

    def step_simulation(self, sim_time_budget: float):
        """
        Advance the simulator by up to sim_time_budget seconds, in chunks of self.dt
        """
        # sample a random action
        # self.data.ctrl[:] = np.random.uniform(low=-1.0, high=1.0, size=self.model.nu)
        
        # safety: ensure we don't iterate forever if dt is nonsensical
        if self.dt <= 0:
            return
        steps = int(sim_time_budget / self.dt)
        # clamp the number of steps to avoid stalling (e.g., if the window was backgrounded)
        max_steps = 5000
        if steps > max_steps:
            steps = max_steps
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def render_once(self):
        # update MuJoCo visualization scene
        mujoco.mjv_updateScene(self.model, self.data, self.scene_option, None, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)

        self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_render(self.viewport, self.scene, self.context)

        if not self.hide_menu:
            # overlay lines (top-left / bottom-right)
            # a small spacing between lines handled by mjr_overlay
            for gridpos, [t1, t2] in self.overlay.items():
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_SHADOW, gridpos, self.viewport, t1, t2, self.context)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def run(self):
        """
        Main loop: simulate in real time (scaled by run_speed_factor) and render.
        """
        last_time = time.time()
        frame_count = 0
        fps_start = last_time
        while not glfw.window_should_close(self.window):
            now = time.time()
            real_dt = now - last_time
            last_time = now

            # compute how much simulation time we should advance this frame
            if not self.paused:
                sim_time_to_advance = real_dt * self.run_speed_factor
                # advance the simulator in discrete steps of self.dt
                self.step_simulation(sim_time_to_advance)

            # update camera settings if mode changed
            self.set_camera()

            # draw
            self.create_overlay(fps=1.0 / max(1e-9, real_dt))
            self.render_once()

            frame_count += 1
            # simple FPS print every second (optional)
            if now - fps_start >= 1.0:
                fps = frame_count / (now - fps_start)
                # keep overlay updated with smooth FPS value next frame
                frame_count = 0
                fps_start = now

        self.close()

    def close(self):
        glfw.destroy_window(self.window)
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description="MuJoCo MJCF (XML) viewer")
    parser.add_argument("xml", type=str, help="Path to MJCF XML file to open (e.g. ant.xml)")
    args = parser.parse_args()

    xml_path = args.xml

    # load model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # create viewer and run
    viewer = MujocoViewer(model, data)
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("Interrupted, closing.")
        viewer.close()


if __name__ == "__main__":
    main()
