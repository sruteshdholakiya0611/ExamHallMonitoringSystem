import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw


class CustomFont:

    def __init__(self, frame_):
        self.frame_ = frame_

    def image_fromArray(self):
        rgb_image = cv2.cvtColor(self.frame_, cv2.COLOR_RGB2BGR)
        return Image.fromarray(rgb_image)

    def singleline_text(self, x, y,
                        text_, text_color,
                        fonts_style, fonts_size,
                        text_bg_color="#F0F8FF",
                        rectangle_outline="#D1D1D1",
                        rectangle_radius=5):

        pillow_image = self.image_fromArray()

        image_draw = ImageDraw.Draw(pillow_image)

        fonts = ImageFont.truetype(font=fonts_style, size=fonts_size)

        left_, top_, right_, bottom_ = image_draw.textbbox((x, y), text_, font=fonts)

        image_draw.rounded_rectangle((left_ - 10, top_ - 10, right_ + 10, bottom_ + 10),
                                     fill=text_bg_color, outline=rectangle_outline, radius=rectangle_radius)

        image_draw.text((x, y), text_, font=fonts, fill=text_color)

        frame_ = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

        return frame_

    def multiline_text(self, x, y,
                       text_, text_color, fonts_style, fonts_size,
                       text_height_spacing=20, text_align='left',
                       text_bg_color="#F0F8FF",
                       rectangle_outline="#D1D1D1",
                       rectangle_radius=5):
        pillow_image = self.image_fromArray()

        image_draw = ImageDraw.Draw(pillow_image)

        fonts = ImageFont.truetype(font=fonts_style, size=fonts_size)

        left_, top_, right_, bottom_ = image_draw.multiline_textbbox((x, y), text_, font=fonts,
                                                                     spacing=text_height_spacing, align=text_align)

        image_draw.rounded_rectangle((left_ - 10, top_ - 10, right_ + 10, bottom_ + 10),
                                     fill=text_bg_color, outline=rectangle_outline, radius=rectangle_radius)

        # alignments = ["left", "center", "right"]

        image_draw.multiline_text((x, y), text_, font=fonts, fill=text_color,
                                  spacing=text_height_spacing, align=text_align)

        frame_ = cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR)

        return frame_

# if __name__ == '__main__':
#     obj = CustomFont(rgb_image=)
#
#     obj.custom_font(x=, y=, fonts_size=, fonts_style=, text_color=, text_=)
