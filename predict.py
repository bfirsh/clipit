import cog
from pathlib import Path
from inference import Transformer
from utils import read_image
import cv2
import tempfile
from utils.image_processing import resize_image, normalize_input, denormalize_input
import numpy as np
import clipit


class Predictor(cog.Predictor):
    def setup(self):
        clipit.reset_settings()

    @cog.input("prompts", type=str, help="Text prompts")
    @cog.input("quality", type=str, options=["draft", "normal", "better", "best"], default="normal", help="quality")
    @cog.input("aspect", type=str, options=["widescreen", "square"], default="widescreen", help="widescreen or square aspect")
    def predict(self, prompts, quality='normal', aspect="widescreen"):
        clipit.add_settings(prompts=prompts, aspect=aspect, quality=quality)
        settings = clipit.apply_settings()
        clipit.do_init(settings)
        clipit.do_run(settings)

        transformer = Transformer(model)
        img = read_image(str(image))
        anime_img = transformer.transform(resize_image(img))[0]
        anime_img = denormalize_input(anime_img, dtype=np.int16)
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        cv2.imwrite(str(out_path), anime_img[..., ::-1])
        return out_path

