import sys
import six
import os
import math

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

# Zooming constants
ZOOM_IN_FACTOR = 1.2
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR

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

        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, '..', 'resources') # todo broken
        print(resource_path)
        # pyglet.resource.path = ['../resources']
        pyglet.resource.path = [resource_path]
        pyglet.resource.reindex()

        self.ground = pyglet.resource.image('wbs_texture_05_resized.jpg')
        self.green = pyglet.resource.image('wbs_texture_05_resized_green.jpg')
        self.red = pyglet.resource.image('wbs_texture_05_resized_red.jpg')

        self.batch = pyglet.graphics.Batch()
        self.face_img = pyglet.resource.image('straight-face.png')
        self.face = pyglet.sprite.Sprite(self.face_img, batch=self.batch)
        self.padding = 1

        self.tile_dim = self.ground.width + self.padding

        # todo must accommodate for the bigger dimension but also check smaller dimension so that it fits.
        # larger dimension check
        ind = np.argmax((self.env.x_max, self.env.y_max))
        larger_grid_dimension = np.max((self.env.x_max, self.env.y_max))
        if ind == 0:
            larger_pixel_dimension = width
        elif ind == 1:
            larger_pixel_dimension = height

        how_many_tiles_you_can_fit_in_larger_dim = math.floor(larger_pixel_dimension / self.tile_dim)
        self.zoom_level = larger_grid_dimension / how_many_tiles_you_can_fit_in_larger_dim # + 5
        # smaller dimension check
        ind = np.argmin([self.env.x_max, self.env.y_max])
        smaller_grid_dimension = np.min([self.env.x_max, self.env.y_max])
        if ind == 0:
            smaller_pixel_dimension = width
        elif ind == 1:
            smaller_pixel_dimension = height
        #
        how_many_tiles_you_can_fit_in_smaller_dim = math.floor(smaller_pixel_dimension / self.tile_dim)
        other_zoom_level = smaller_grid_dimension / how_many_tiles_you_can_fit_in_smaller_dim # + 5

        if other_zoom_level > self.zoom_level:
            self.zoom_level = other_zoom_level

        if how_many_tiles_you_can_fit_in_larger_dim > larger_grid_dimension and how_many_tiles_you_can_fit_in_smaller_dim > smaller_grid_dimension:
            self.zoom_level = 1

        self.zoomed_width = width * self.zoom_level
        self.zoomed_height = height * self.zoom_level

        self.left = 0
        self.right = self.zoomed_width
        self.bottom = 0
        self.top = self.zoomed_height

        print('tile_dim: {}. grid_shape: {}, how_many_tiles_you_can_fit_in_larger_dim: {}'.format(
            self.tile_dim, [self.env.x_max, self.env.y_max], how_many_tiles_you_can_fit_in_larger_dim))
        print('zoom:', self.zoom_level)
        print('width: {}, height: {}, zoomed_width: {}, zoomed_height: {}'.format(width, height, self.zoomed_width, self.zoomed_height))

        self.geoms = []
        self.onetime_geoms = []
        self.transform = Transform()

        # glScalef(2.0, 2.0, 2.0)
        # glScalef(0.5, 0.5, 0.5)

        glViewport(0, 0, width, height)
        # glViewport(0, 0, self.zoomed_width, self.zoomed_height)
        # Initialize camera values
        # self.left = 0
        # self.right = width
        # self.bottom = 0
        # self.top = height
        # self.zoom_level = 1
        # self.zoomed_width = width
        # self.zoomed_height = height

        # Set antialiasing
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

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
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix() # todo causes stack overflow after resize or always?

        glClearColor(0, 0, 0, 1)
        glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        # self.transform.enable()
        # for geom in self.geoms:
        #     geom.render()
        # for geom in self.onetime_geoms:
        #     geom.render()
        # self.transform.disable()

        # self.batch.draw()

        # self.ground.blit(self.face.x, self.face.y)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # have to flip pixel location. top-left is initial state = x, y = 0, 0 = state 0
        pix_grid_height = (self.env.y_max - 1) * (self.ground.height + self.padding)

        for i, (x, y) in enumerate(self.env.world):
            x_pix_loc, y_pix_loc = x * (self.ground.width + self.padding), pix_grid_height - y * (self.ground.height + self.padding)
            if self.env.is_terminal(i): # if terminal
                self.green.blit(x_pix_loc, y_pix_loc)
            # elif i == 3: # lava
            #     self.red.blit(x_pix_loc, y_pix_loc)
            elif not self.env._is_wall(i): # todo totally wrong inverse?
                self.red.blit(x_pix_loc, y_pix_loc)
            else:
                self.ground.blit(x_pix_loc, y_pix_loc)

        self.face.x = self.env.world[self.env.current_state][0] * (self.ground.width + self.padding)
        self.face.y = pix_grid_height - self.env.world[self.env.current_state][1] * (self.ground.height + self.padding)
        # print('x, y: {}, {}. x, y pixels: {}, {}'.format(self.env.world[self.env.current_state][0], self.env.world[self.env.current_state][1],
        #                                                  self.face.x, self.face.y))

        self.face.draw()

        # glBegin(GL_QUADS)
        # glColor3ub(0xFF, 0, 0)
        # glVertex2i(10, 10)
        #
        # glColor3ub(0xFF, 0xFF, 0)
        # glVertex2i(110, 10)
        #
        # glColor3ub(0, 0xFF, 0)
        # glVertex2i(110, 110)
        #
        # glColor3ub(0, 0, 0xFF)
        # glVertex2i(10, 110)
        # glEnd()
        glPopMatrix()

        # dx, dy = 1, 1
        # self.left -= dx * self.zoom_level
        # self.right -= dx * self.zoom_level
        # self.bottom -= dy * self.zoom_level
        # self.top -= dy * self.zoom_level

        # self.zoom_level += 1
        # self.zoomed_width *= 1
        # self.zoomed_height *= 1
        #
        # self.left = mouse_x_in_world - mouse_x * self.zoomed_width
        # self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
        # self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
        # self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

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
