import cv2
import numpy as np
from PIL import Image, ImageDraw


class CustomFrame:
    def __init__(self, frame_):
        self.frame_ = frame_

    def rectangle_frame(self, rectangle_box, length=25, line_width=4, rectangle_width=1):
        left_, top_, right_, bottom_ = rectangle_box
        x1, y1 = left_ + right_, top_ + bottom_

        rgb_image = cv2.cvtColor(self.frame_, cv2.COLOR_BGR2RGB)

        pillow_image = Image.fromarray(rgb_image)

        image_draw = ImageDraw.Draw(pillow_image)

        image_draw.line(xy=[(left_, top_), (left_ + length, top_)], fill='#6495ED', width=line_width)
        image_draw.line(xy=[(left_, top_), (left_, top_ + length)], fill='#6495ED', width=line_width)

        image_draw.line(xy=[(x1, top_), (x1 - length, top_)], fill='#6495ED', width=line_width)
        image_draw.line(xy=[(x1, top_), (x1, top_ + length)], fill='#6495ED', width=line_width)

        image_draw.line(xy=[(left_, y1), (left_ + length, y1)], fill='#6495ED', width=line_width)
        image_draw.line(xy=[(left_, y1), (left_, y1 - length)], fill='#6495ED', width=line_width)

        image_draw.line(xy=[(x1, y1), (x1 - length, y1)], fill='#6495ED', width=line_width)
        image_draw.line(xy=[(x1, y1), (x1, y1 - length)], fill='#6495ED', width=line_width)

        image_draw.rounded_rectangle(xy=[left_, top_, x1, y1], outline='#6495ED',
                                     radius=10, width=rectangle_width)

        frame_ = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

        return frame_
