import os
import cv2
from matplotlib import colormaps


class Anim:

    def __init__(
            self, 
            name, 
            dpi, 
            aspect_ratio, 
            channels="RGB", 
            cmap="inferno",
            fps=60
        ) -> None:
        self.channels = 3 if channels == "RGB" else 1
        self.fps = fps
        self.h = int(dpi * aspect_ratio[0])
        self.w = int(dpi * aspect_ratio[1])
        self.cmap = colormaps[cmap]
        
        filename = os.path.join(os.path.dirname(__file__), name + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, fourcc, fps, (self.w,self.h))
        return 

    def step(self, frame):
        if frame.shape != (self.h, self.w):
            frame = frame.repeat(self.h//frame.shape[0], axis=0)
            frame = frame.repeat(self.w//frame.shape[1], axis=1)
        if frame.shape != (self.h, self.w):
            frame = cv2.resize(frame, (self.w, self.h)) 
            # frame = np.resize(frame, (self.h, self.w, 3))             
        self.out.write(frame)

    def map_colors(self, frame, max):
        frame = frame/max+1
        frame[ 0, 0] = 0
        # frame[-1,-1] = 1
        frame = self.cmap(frame)[..., :-1]
        frame = frame[..., ::-1]
        frame = (frame*255).astype("uint8")
        return frame

    def close(self):
        self.out.release()
        cv2.destroyAllWindows()

    def __del__(self):
        self.close()
        