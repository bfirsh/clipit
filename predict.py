import cog
from pathlib import Path
import cv2
import tempfile
import clipit


class Predictor(cog.Predictor):
    def setup(self):
        clipit.reset_settings()

    @cog.input("prompts", type=str, help="Text prompts")
    @cog.input("quality", type=str, options=["draft", "normal", "better", "best"], default="normal", help="quality")
    @cog.input("aspect", type=str, options=["widescreen", "square"], default="widescreen", help="widescreen or square aspect")
    def predict(self, prompts, quality='normal', aspect="widescreen"):
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        clipit.add_settings(prompts=prompts, aspect=aspect, quality=quality, output=str(out_path))
        settings = clipit.apply_settings()
        clipit.do_init(settings)
        clipit.do_run(settings)
        return out_path

