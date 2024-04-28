import os
import cv2
from matplotlib import colormaps


class Anim:

    def __init__(self, name, dpi, aspect_ratio, cmap="inferno", fps=60):
        self.fps = fps
        self.cmap = colormaps[cmap]
        self.h = int(dpi * aspect_ratio[0])
        self.w = int(dpi * aspect_ratio[1])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = os.path.join(os.path.dirname(__file__), "videos", name + '.mp4')
        self.encoder = cv2.VideoWriter(filename, fourcc, fps, (self.w,self.h))
        return 

    def step(self, frame):
        ## SCALE WITHOUT INTERPOLATION
        if frame.shape != (self.h, self.w):
            frame = frame.repeat(self.h//frame.shape[0], axis=0)
            frame = frame.repeat(self.w//frame.shape[1], axis=1)
        ## SCALE WITH INTERPOLATION
        if frame.shape != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h))       
        self.encoder.write(frame)

    def map_colors(self, frame, max):
        ## SET MIN AND MAX SO THAT CMAP DOESNT RENORMALIZE
        frame = frame / max + 1
        frame[0,0] = 0
        frame[-1,-1] = 1
        frame = self.cmap(frame)[..., :-1]
        frame = frame[..., ::-1]
        ## RESET PIXELS TO LOCAL VALUE
        frame[0,0] = frame[0,1]
        frame[-1,-1] = frame[-1,-2]
        frame = (frame*255).astype("uint8")
        return frame

    def close(self):
        self.encoder.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.close()
        