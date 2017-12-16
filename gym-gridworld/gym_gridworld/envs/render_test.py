import sys


import pyglet

pyglet.resource.path = ['../resources']
pyglet.resource.reindex()

window = pyglet.window.Window()
label = pyglet.text.Label('Hello, world',
                          font_name='Times New Roman',
                          font_size=36,
                          x=window.width//2, y=window.height//2,
                          anchor_x='center', anchor_y='center')




sprite = pyglet.resource.image('sprite.png')
width, height = 400, 200
# w = min(image.height, height)/max(image.height, height)
# h = max(min(width, image.width)/max(width, image.width)
# image.scale = w, h
sprite.scale = 200, 300
# image.scale = min(image.height, height)/max(image.height, height), max(min(width, image.width)/max(width, image.width))
# image.scale = min(image.height, height)/max(image.height, height), max(min(width, image.width)/max(width, image.width))
# Usually not needed, and should not be tampered with,
# but for a various bugs when using sprite-inheritance on a user-defined
# class, these values will need to be updated manually:
sprite.width = width
sprite.height = height
sprite.texture.width = width
sprite.texture.height = height

# @window.event
# def on_key_press(symbol, modifiers):
#     print('A key was pressed')

from pyglet.window import key

@window.event
def on_key_press(symbol, modifiers):
    if symbol == key.A:
        print('The "A" key was pressed.')
    elif symbol == key.LEFT:
        print('The left arrow key was pressed.')
    elif symbol == key.ENTER:
        print('The enter key was pressed.')



from pyglet.window import mouse

@window.event
def on_mouse_press(x, y, button, modifiers):
    if button == mouse.LEFT:
        print('The left mouse button was pressed.')

# window.push_handlers(pyglet.window.event.WindowEventLogger()) # for a lot of output and all events logged

import numpy as np
world = np.fromiter(((x, y) for x in np.nditer(np.arange(4))
                             for y in np.nditer(np.arange(4))), dtype='int16, int16')
current_state = 5

batch = pyglet.graphics.Batch()
cur_x, cur_y = world[5]


ground = pyglet.resource.image('wbs_texture_05_resized.jpg')
red = pyglet.resource.image('wbs_texture_05_resized_red.jpg')
green = pyglet.resource.image('wbs_texture_05_resized_green.jpg')
face_img = pyglet.resource.image('straight-face.png')
face = pyglet.sprite.Sprite(face_img, x=cur_x * ground.width, y=cur_y * ground.height, batch=batch)
padding = 1
print (ground.width, ground.height)





@window.event
def on_draw():
    window.clear()
    label.draw()

    #window.clear()

    for i, (x, y) in enumerate(world):
        if i == len(world) - 1:
            green.blit(x * (ground.width + padding), y * (ground.height + padding))
        elif i == 3:
            red.blit(x * (ground.width + padding), y * (ground.height + padding))
        else:
            ground.blit(x * (ground.width + padding), y * (ground.height + padding))

        #if i == current_state:
        # face.blit(x * (ground.width + padding), y * (ground.height + padding))
        batch.draw()

pyglet.app.run()











sys.exit()

# http://pyglet.readthedocs.io/en/pyglet-1.3-maintenance/programming_guide/quickstart.html
# http://steveasleep.com/pyglettutorial.html

pyglet.resource.path = ['../resources']
pyglet.resource.reindex()

game_window = pyglet.window.Window(800, 600)

# player_image = pyglet.resource.image("sprite.png")
player_image = pyglet.resource.image("G.jpg")
# bullet_image = pyglet.resource.image("bullet.png")
# asteroid_image = pyglet.resource.image("asteroid.png")
#
score_label = pyglet.text.Label(text="Score: 0", x=10, y=575)
level_label = pyglet.text.Label(text="My Amazing Game", x=400, y=575, anchor_x='center')



@game_window.event
def on_draw():
    game_window.clear()
    level_label.draw()
    score_label.draw()
    center_image(player_image)
    # player_image.draw()

def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width/2
    image.anchor_y = image.height/2

center_image(player_image)

if __name__ == '__main__':

    pyglet.app.run()
    center_image(player_image)
