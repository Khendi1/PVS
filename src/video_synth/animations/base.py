from abc import ABC, abstractmethod
import logging
import numpy as np

log = logging.getLogger(__name__)

_FULLSCREEN_QUAD_VERT = """
#version 330
in vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_QUAD_VERTS = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')


class Animation(ABC):
    """
    Abstract class to help unify animation frame retrieval
    """
    def __init__(self, params, width=640, height=480, group=None):
        self.params = params
        self.width = width
        self.height = height
        self.group = group


    @abstractmethod
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        """
        """
        raise NotImplementedError("subgroupes should implement this method.")


class GLSLAnimation(Animation):
    """
    Base class for GPU-accelerated animations using ModernGL standalone context.

    Subclasses create one or more GLSL fragment shader programs via
    ``_compile_program(fragment_src)`` and render them each frame using
    ``_render(program, uniforms)``.  The ModernGL context and FBO are managed
    here so subclasses don't repeat boilerplate.

    Falls back gracefully to a black frame when ModernGL is unavailable (e.g.
    in headless CI environments without a GPU).

    Example subclass::

        class MyGPUAnim(GLSLAnimation):
            def __init__(self, params, width=640, height=480, group=None):
                super().__init__(params, width, height, group)
                self._prog = self._compile_program(MY_FRAG_SRC)

            def get_frame(self, frame=None):
                return self._render(self._prog, {
                    'u_resolution': (self.width, self.height),
                    'u_time': time.time(),
                })
    """

    def __init__(self, params, width=640, height=480, group=None):
        super().__init__(params, width, height, group)
        self._ctx = None
        self._vbo = None
        self._fbo = None
        self._fbo_texture = None
        self._gpu_ok = False

        try:
            import moderngl
            import cv2 as _cv2
            self._moderngl = moderngl
            self._cv2 = _cv2
            self._ctx = moderngl.create_context(standalone=True)
            self._vbo = self._ctx.buffer(_QUAD_VERTS)
            self._fbo_texture = self._ctx.texture((width, height), 3)
            self._fbo = self._ctx.framebuffer(color_attachments=[self._fbo_texture])
            self._gpu_ok = True
        except Exception as exc:
            log.warning("GLSLAnimation: GPU context unavailable (%s); falling back to black frames.", exc)

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _compile_program(self, fragment_src: str):
        """Compile a fragment shader and return a (program, vao) tuple.

        Returns ``None`` when GPU is not available.
        """
        if not self._gpu_ok:
            return None
        try:
            prog = self._ctx.program(
                vertex_shader=_FULLSCREEN_QUAD_VERT,
                fragment_shader=fragment_src,
            )
            vao = self._ctx.vertex_array(prog, [(self._vbo, '2f', 'in_vert')])
            return (prog, vao)
        except Exception as exc:
            log.error("GLSLAnimation: shader compile failed: %s", exc)
            return None

    def _render(self, program_vao, uniforms: dict) -> np.ndarray:
        """Render a fullscreen quad with *uniforms* and return a BGR ndarray.

        *program_vao* is the tuple returned by ``_compile_program``.
        Returns a black frame when GPU is unavailable or *program_vao* is None.
        """
        if not self._gpu_ok or program_vao is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        prog, vao = program_vao
        for name, value in uniforms.items():
            if name in prog:
                prog[name].value = value

        self._fbo.use()
        self._ctx.clear(0.0, 0.0, 0.0)
        vao.render(self._moderngl.TRIANGLE_STRIP)

        data = np.frombuffer(self._fbo.read(components=3), dtype=np.uint8)
        img = data.reshape((self.height, self.width, 3))
        return self._cv2.flip(img, 0)  # OpenGL y-axis is flipped vs OpenCV

    @abstractmethod
    def get_frame(self, frame: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError
