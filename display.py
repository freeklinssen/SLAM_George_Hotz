import sdl2
import sdl2.ext

class display2d():
    def __init__(self, w, h):
        sdl2.ext.init()
        self.w, self.h = w, h
        self.window = sdl2.ext.Window("line detection", size=(w, h))
        self.window.show()

    def show(self, img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = img.swapaxes(0, 1)

        self.window.refresh()


