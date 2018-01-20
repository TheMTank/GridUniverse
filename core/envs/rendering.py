import sys
import six
import os
import math
import time
import random

import pyglet

import numpy as np
from gym import error
from gym.utils import reraise

import math

try:
    import pyglet
except ImportError as e:
    reraise(suffix="HINT: you can install pyglet directly via 'pip install pyglet'. But if you really just want to install all Gym dependencies and not have to think about it, 'pip install -e .[all]' or 'pip install gym[all]' will do it.")

try:
    from pyglet.gl import *
except ImportError as e:
    reraise(prefix="Error occured while running `from pyglet.gl import *`",suffix="HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. If you're running on a server, you may need a virtual frame buffer; something like this should work: 'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'")

RAD2DEG = 57.29577951308232

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):
    def __init__(self, env, width, height, display=None):
        display = get_display(display)
        self.env = env

        self.width = width
        self.height = height
        self.window = pyglet.window.Window(width=width, height=height, display=display, resizable=True) # todo resizable remove or keep
        self.window.on_close = self.window_closed_by_user
        self.FPS = 0

        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, '..', 'resources')
        pyglet.resource.path = [resource_path]
        pyglet.resource.reindex()

        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        self.face_img = pyglet.resource.image('straight-face.png')
        self.ground_img = pyglet.resource.image('wbs_texture_05_resized.jpg')
        self.terminal_goal_img = pyglet.resource.image('wbs_texture_05_resized_green.jpg')
        self.wall_img = pyglet.resource.image('wbs_texture_05_resized_red.jpg')

        self.padding = 1
        self.tile_dim = self.ground_img.width + self.padding

        self.wall_sprites = []
        self.terminal_sprites = []
        self.ground_sprites = []

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)
        self.face = pyglet.sprite.Sprite(self.face_img, batch=self.batch, group=foreground)

        # Lets say I want to fit 5 tiles + x_max tiles + 5 tiles always on the x dimension
        # Therefore start x should be 5 tiles * tile_dim
        # Therefore larger_grid_dimension is x_max + 10 tiles
        # Therefore larger_pixel_dimension is width
        # Therefore how_many_tiles_you_can_fit_in_larger_dim = math.floor(larger_pixel_dimension / self.tile_dim)
        # Therefore self.zoom_level = larger_grid_dimension / how_many_tiles_you_can_fit_in_larger_dim
        # Therefore same for y_level

        # Don't use extra tiles. Just center it.
        # Larger dimension by num tiles should just fit into screen with z pixels on each side

        # self.num_extra_tiles = 10
        self.num_extra_tiles = 4

        # must accommodate for the bigger dimension but also check smaller dimension so that it fits.
        num_tiles_to_fit_in_width = math.floor(width / self.tile_dim)
        zoom_level_for_width = (self.env.x_max + self.num_extra_tiles) / num_tiles_to_fit_in_width
        pixel_width_of_grid = math.floor(zoom_level_for_width * width) # why width?

        num_tiles_to_fit_in_height = math.floor(height / self.tile_dim)
        zoom_level_for_height = (self.env.y_max + self.num_extra_tiles) / num_tiles_to_fit_in_height
        pixel_height_of_grid = math.floor(zoom_level_for_height * height)

        self.zoom_level = np.max((zoom_level_for_width, zoom_level_for_height))

        print('zoom_level_for_width: {}, zoom_level_for_height: {}, zoom_level: {}'.format(zoom_level_for_width, zoom_level_for_height, self.zoom_level))
        print('pixel_width_of_grid: {}, pixel_height_of_grid: {}'.format(pixel_width_of_grid, pixel_height_of_grid))

        self.zoomed_width = width * self.zoom_level
        self.zoomed_height = height * self.zoom_level

        # because width is always the bigger dimension we will always move the grid in the x
        self.x_distance_to_move = self.zoomed_width / 2 - pixel_width_of_grid / 2

        self.left = 0
        self.right = self.zoomed_width
        self.bottom = 0
        self.top = self.zoomed_height

        print('x_distance_to_move:', self.x_distance_to_move)
        print('tile_dim: {}. grid_shape: {}'.format(self.tile_dim, [self.env.x_max, self.env.y_max]))
        # print('how_many_tiles_you_can_fit_in_smaller_dim: {}, how_many_tiles_you_can_fit_in_larger_dim: {}'.format(how_many_tiles_you_can_fit_in_smaller_dim, how_many_tiles_you_can_fit_in_larger_dim))
        print('zoom:', round(self.zoom_level, 4))
        print('width: {}, height: {}, zoomed_width: {}, zoomed_height: {}'.format(width, height, self.zoomed_width, self.zoomed_height))

        # todo create function to calculate x and y pixel position from grid position for face, arrows and agent path
        # have to flip pixel location. top-left is initial state = x, y = 0, 0 = state 0
        self.pix_grid_height = (self.env.y_max) * self.tile_dim + (self.num_extra_tiles // 2) * self.tile_dim

        for i, (x, y) in enumerate(self.env.world):
            x_pix_loc = x * self.tile_dim + self.x_distance_to_move + (self.num_extra_tiles // 2) * self.tile_dim
            y_pix_loc = self.pix_grid_height - (y * self.tile_dim)
            if self.env.is_terminal(i):  # if terminal
                self.wall_sprites.append(
                    pyglet.sprite.Sprite(self.terminal_goal_img, x=x_pix_loc, y=y_pix_loc, batch=self.batch, group=background))
            elif self.env._is_wall(i):
                self.wall_sprites.append(
                    pyglet.sprite.Sprite(self.wall_img, x=x_pix_loc, y=y_pix_loc, batch=self.batch, group=background))
            else:
                self.wall_sprites.append(
                    pyglet.sprite.Sprite(self.ground_img, x=x_pix_loc, y=y_pix_loc, batch=self.batch, group=background))

        glViewport(0, 0, width, height)

        # Set antialiasing
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # cart = make_circle(500)
        # cart.set_color(255, 0, 0)
        # self.carttrans = Transform()
        # cart.add_attr(self.carttrans)
        # self.add_geom(cart)

        # self.line = Line((0, 0), (500, 500))
        # self.add_geom(self.line)
        # line.

    # def calculate_policy_lines(self, policy_arrows_map): # todo or policy map?
    def render_policy_arrows(self, policy): # todo or policy map?
        # todo show policy probabilities as well?
        # remove all previous arrows and recalculate
        self.geoms = [] # todo only remove arrows

        unicode_arrows = np.array([u'\u2191', u'\u2192', u'\u2193', u'\u2190'  # up, right, down, left
                                                                    u'\u2194', u'\u2195'],
                                  dtype='<U1')  # left-right, up-down
        policy_arrows_map = np.empty(policy.shape[0], dtype='<U4')
        # probs_for_state = np.zeros(policy.shape)
        for idx, state in enumerate(np.nditer(np.arange(policy.shape[0]))):
            # probs_for_state[idx] = []
            # for action in range(4): #self.env.action_space:
            #     probs_for_state[idx].append(policy[state][action])
            # probs_for_state[state] = policy_state
            # find index of actions where the probability is > 0
            optimal_actions = np.where(np.around(policy[state], 8) > np.around(np.float64(0), 8))[0]
            # match actions to unicode values of the arrows to be displayed
            for action in optimal_actions:
                policy_arrows_map[state] = np.core.defchararray.add(policy_arrows_map[state], unicode_arrows[action])
        policy_probabilities = np.fromiter((policy[state] for state in np.nditer(np.arange(policy.shape[0]))),
                                           dtype='float64, float64, float64, float64')


        for state_index, (x, y) in enumerate(self.env.world):
            x_pix_loc, y_pix_loc = x * self.tile_dim, self.pix_grid_height - y * self.tile_dim
            if self.env.is_terminal(state_index):  # if terminal
                pass
            elif self.env._is_wall(state_index):
                pass
            else:
                center = np.array([x_pix_loc + self.tile_dim / 2, y_pix_loc + self.tile_dim / 2]).astype(int)
                arrow_base_length_full_prob = 20
                arrow_width = 5
                arrow_height = 5
                # unicode_arrows = np.array([u'\u2191', u'\u2192', u'\u2193', u'\u2190'  # up, right, down, left
                #                                                             u'\u2194', u'\u2195'],
                #                           dtype='<U1')  # left-right, up-down
                # policy_arrows_map[state_index] = np.core.defchararray.add(policy_arrows_map[state_index], u'\u2193')
                # policy_arrows_map[state_index] += u'\u2193'
                # print(policy_arrows_map[state_index])
                # for char in policy_arrows_map[state_index]:
                # for key in probs_for_state.keys():
                # for action_idx, probability in enumerate(probs_for_state[state]):
                #     i = 0
                    # for action_idx, probability in enumerate(probs_for_state[key]):
                print(policy[state_index])
                for action_idx, probability in enumerate(policy[state_index]):
                        # direction, towards, arrow_head_vertices = None, None, None
                        # print(action_idx, probability)
                        # print(probs_for_state[key])
                        arrow_base_length = round(probability * arrow_base_length_full_prob)
                        # if arrow_base_length < 3 or probability < 0.1:
                        if probability < 0.1:
                            # print('arrow base length too small')
                            continue
                            break
                        # print(policy_arrows_map)
                        # print(policy_arrows_map[state_index])
                        # if len(policy_arrows_map[state_index]) > 1:
                        #     print('WAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAARNING')
                        #     for ch in policy_arrows_map[state_index]:
                        #         print(ch)
                        #     print()

                        # if char == u'\u2191': # up
                        if action_idx == 0: # up
                            # np because of vector math
                            # i += 1
                            # print(i, ' times here for same action')
                            # assert i == 1
                            direction = np.array([0, arrow_base_length]).astype(int)
                            towards = center + direction
                            arrow_head_vertices = [center + direction + np.array([arrow_width, 0]), # arrow head bottom right vertex
                                                   center + direction + np.array([-arrow_width, 0]), # arrow head bottom left vertex
                                                   center + direction + np.array([0, arrow_height])] # arrow pointy bit
                        # elif char == u'\u2193': # down
                        elif action_idx == 2: # down
                            print('arrow_base_length:', arrow_base_length)
                            direction = np.array([0, - arrow_base_length]).astype(int)
                            towards = center + direction
                            arrow_head_vertices = [center + direction + np.array([arrow_width, 0]),
                                                   center + direction + np.array([-arrow_width, 0]),
                                                   center + direction + np.array([0, -arrow_height])]
                        # # elif char == u'\u2192': # right
                        elif action_idx == 1: # right
                            # continue
                            direction = np.array([0 + arrow_base_length, 0]).astype(int)
                            towards = center + direction
                            arrow_head_vertices = [center + direction + np.array([0, -arrow_width]),
                                                   center + direction + np.array([0, +arrow_width]),
                                                   center + direction + np.array([arrow_height, 0])]
                        elif action_idx == 3:  # left
                            # continue
                            direction = np.array([0 - arrow_base_length, 0]).astype(int)
                            towards = center + direction
                            arrow_head_vertices = [center + direction + np.array([0, +arrow_width]),
                                                   center + direction + np.array([0, -arrow_width]),
                                                   center + direction + np.array([-arrow_height, 0])]

                        if arrow_base_length < 3 or probability < 0.1:
                            print('SOMETHING WRONG!!!')
                            break

                        center = tuple(center)
                        towards = tuple(towards)
                        arrow_head_vertices = [tuple(a_h_v) for a_h_v in arrow_head_vertices]
                        line = Line(center, towards)
                        arrow_head = FilledPolygon(arrow_head_vertices)
                        self.add_geom(arrow_head)
                        self.add_geom(line)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width/(right-left)
        scaley = self.height/(top-bottom)
        self.transform = Transform(
            translation=(-left*scalex, -bottom*scaley),
            scale=(scalex, scaley))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    # def render(self, return_rgb_array=False):
    def render(self, return_rgb_array=False):
        start_time = time.time()
        # todo if close don't crash
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        glClearColor(0, 0, 0, 1)
        glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) # todo needed?
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)


        # Draw text
        font_size = 50
        fps_string = "FPS: {}".format(self.FPS)
        fps_label = pyglet.text.Label(text=fps_string, x=self.zoomed_width - len(fps_string) * font_size, y=self.zoomed_height - 80, font_size=font_size)
        fps_label.draw()

        if hasattr(self.env, 'step_num'):
            # todo get text size to properly align
            step_string = "Step: {}".format(self.env.step_num)

            step_num_label = pyglet.text.Label(text=step_string, x=self.zoomed_width - len(step_string) * font_size, y=self.zoomed_height - 160, font_size=font_size)
            step_num_label.draw()

        # Render agent
        self.face.x = self.env.world[self.env.current_state][0] * self.tile_dim + self.x_distance_to_move + (self.num_extra_tiles // 2) * self.tile_dim
        self.face.y = self.pix_grid_height - self.env.world[self.env.current_state][1] * self.tile_dim
        self.batch.draw()

        # Render all geoms e.g. policy arrows
        # self.draw_circle(500, 30)
        # self.line.end = (self.line.end[0], self.line.end[1] - 1)
        self.transform.enable()
        for geom in self.geoms:
            geom.render()
        for geom in self.onetime_geoms:
            geom.render()
        self.transform.disable()

        # Render agent path
        glBegin(GL_QUADS)
        a = 0.3
        discount = 0.96
        padding = 10
        for i, (x, y) in enumerate(self.env.last_n_states[::-1]):
            x_pix_loc = int(x * self.tile_dim + self.x_distance_to_move + (self.num_extra_tiles // 2) * self.tile_dim)
            y_pix_loc = int(self.pix_grid_height - y * self.tile_dim)

            a *= discount
            # if x_pix_loc == self.face.x and y_pix_loc == self.face.y:
            if x == self.env.world[self.env.current_state][0] and y == self.env.world[self.env.current_state][1]:
                continue

            # todo find best colours
            # glColor4f(0x8, 0x8, 0x8, a)
            glColor4f(0xFF, 0, 0, a)
            # glVertex2i(x_pix_loc + padding, y_pix_loc + padding)
            glVertex2i(x_pix_loc, y_pix_loc)
            # glColor4f(0x8, 0x8, 0x8, a)
            glColor4f(0xFF, 0xFF, 0, a)
            # glVertex2i(x_pix_loc + self.tile_dim - padding, y_pix_loc + padding)
            glVertex2i(x_pix_loc + self.tile_dim, y_pix_loc)
            # glColor4f(0x8, 0x8, 0x8, a)
            glColor4f(0, 0xFF, 0, a)
            # glVertex2i(x_pix_loc + self.tile_dim - padding, y_pix_loc + self.tile_dim - padding)
            glVertex2i(x_pix_loc + self.tile_dim, y_pix_loc + self.tile_dim)
            # glColor4f(0x8, 0x8, 0x8, a)
            glColor4f(0, 0, 0xFF, a)
            # glVertex2i(x_pix_loc + padding, y_pix_loc + self.tile_dim - padding)
            glVertex2i(x_pix_loc, y_pix_loc + self.tile_dim)
        glEnd()

        glPopMatrix()

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]
        self.window.flip()
        self.onetime_geoms = []


        self.FPS = math.floor(1 / (time.time() - start_time))
        # print('Time taken for render: {}, {} FPS'.format(round(time.time() - start_time, 4), self.FPS))
        return arr

    # Convenience
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

def _add_attrs(geom, attrs):
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])

class Geom(object):
    def __init__(self):
        self._color=Color((0, 0, 0, 1.0))
        self.attrs = [self._color]
    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()
    def render1(self):
        raise NotImplementedError
    def add_attr(self, attr):
        self.attrs.append(attr)
    def set_color(self, r, g, b):
        self._color.vec4 = (r, g, b, 1)

class Attr(object):
    def enable(self):
        raise NotImplementedError
    def disable(self):
        pass

class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1,1)):
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)
    def enable(self):
        glPushMatrix()
        glTranslatef(self.translation[0], self.translation[1], 0) # translate to GL loc ppint
        glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        glScalef(self.scale[0], self.scale[1], 1)
    def disable(self):
        glPopMatrix()
    def set_translation(self, newx, newy):
        self.translation = (float(newx), float(newy))
    def set_rotation(self, new):
        self.rotation = float(new)
    def set_scale(self, newx, newy):
        self.scale = (float(newx), float(newy))

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4
    def enable(self):
        glColor4f(*self.vec4)

class LineStyle(Attr):
    def __init__(self, style):
        self.style = style
    def enable(self):
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(1, self.style)
    def disable(self):
        glDisable(GL_LINE_STIPPLE)

class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = stroke
    def enable(self):
        glLineWidth(self.stroke)

class Point(Geom):
    def __init__(self):
        Geom.__init__(self)
    def render1(self):
        glBegin(GL_POINTS) # draw point
        glVertex3f(0.0, 0.0, 0.0)
        glEnd()

class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v
    def render1(self):
        if   len(self.v) == 4 : glBegin(GL_QUADS)
        elif len(self.v)  > 4 : glBegin(GL_POLYGON)
        else: glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()

def make_circle(radius=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*radius, math.sin(ang)*radius))
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)

def make_polygon(v, filled=True):
    if filled: return FilledPolygon(v)
    else: return PolyLine(v, True)

def make_polyline(v):
    return PolyLine(v, False)

def make_capsule(length, width):
    l, r, t, b = 0, length, width/2, -width/2
    box = make_polygon([(l,b), (l,t), (r,t), (r,b)])
    circ0 = make_circle(width/2)
    circ1 = make_circle(width/2)
    circ1.add_attr(Transform(translation=(length, 0)))
    geom = Compound([box, circ0, circ1])
    return geom

class Compound(Geom):
    def __init__(self, gs):
        Geom.__init__(self)
        self.gs = gs
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]
    def render1(self):
        for g in self.gs:
            g.render()

class PolyLine(Geom):
    def __init__(self, v, close):
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)
    def render1(self):
        glBegin(GL_LINE_LOOP if self.close else GL_LINE_STRIP)
        for p in self.v:
            glVertex3f(p[0], p[1],0)  # draw each vertex
        glEnd()
    def set_linewidth(self, x):
        self.linewidth.stroke = x

class Line(Geom):
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0)):
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = LineWidth(1)
        self.add_attr(self.linewidth)

    def render1(self):
        glBegin(GL_LINES)
        glVertex2f(*self.start)
        glVertex2f(*self.end)
        glEnd()

class Image(Geom):
    def __init__(self, fname, width, height):
        Geom.__init__(self)
        self.width = width
        self.height = height
        img = pyglet.image.load(fname)
        self.img = img
        self.flip = False
    def render1(self):
        self.img.blit(-self.width/2, -self.height/2, width=self.width, height=self.height)

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()
