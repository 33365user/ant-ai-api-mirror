from manim import *
from utils import ClassificationRequest
from utils import *
from params import Params
import os, sys
import inspect
import importlib as il
from importlib import util as ilutil
from importlib import abc as ilabc


class ObjectRepresentationScene(Scene):
    def __init__(self, cls: object, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colorrepr = {"key":YELLOW, "value":WHITE}
        self.formatrepr = {"key": {"bold":BOLD, "italic":None, "fontsize":96},
                           "value": {"bold":None, "italic":None, "fontsize":90},
                           }

        self.classtorepresent = cls

    def construct(self):
        params = vars(self.classtorepresent)
        numvars = len(params)
        square = Square(numvars * 1.5 * 2)
        square.set_fill(GREEN, opacity=0.7)
        self.play(Create(square))

    def validate(self, c):
        pass

def repr_it(c: object):
    """
    Converts a class C (assumes instatiated params) to a Manim representation that is understandable visually.

    :param c:
    :return:
    """
    def repr_it_wrapper(f: callable):
        def inner_wrapper(*args, **kwargs):
            return f(*args, **kwargs)
        return inner_wrapper
    return repr_it_wrapper


class TestScene(Scene):
    def construct(self):
        circle = Circle()
        circle.set_fill(GREEN, opacity=0.9)

        square = Square()
        square.set_fill(GREEN, opacity=0.7)
        square.rotate(PI / 4)

        self.play(Create(circle))
        self.play(Transform(circle, square)) # interpolate
        self.play(FadeOut(circle))


if __name__ == "__main__":
    import subprocess
    cwd = os.getcwd()
    fname = os.path.basename(sys.argv[0])
    targetclass = ObjectRepresentationScene(ClassificationRequest("1x2re3gvg43", "path/to/file"))
    cmd = f"manim -pqh {fname} {targetclass.__name__}"
    subprocess.run(cmd.split(" "))
