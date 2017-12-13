import pyglet

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
