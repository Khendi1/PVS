import cv2
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram

# --- GLSL Shader Source Code (Inline) ---
# The shaders remain the same as the rendering logic is OpenGL standard.

VERTEX_SHADER_SOURCE = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

# Fragment Shader: Applies the simple 'Cyberpunk Grayscale' effect
FRAGMENT_SHADER_SOURCE = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ourTexture;
uniform float time; // Time uniform for animation

void main()
{
    vec4 texColor = texture(ourTexture, TexCoord);
    float gray = dot(texColor.rgb, vec3(0.2126, 0.7152, 0.0722));

    // Apply color shift effect (Cyberpunk/Monochrome look)
    vec3 outputColor = vec3(gray, gray, gray);
    outputColor.g *= 1.0 + sin(time) * 0.1;
    outputColor.b *= 1.2 + cos(time) * 0.1;
    outputColor.r *= 0.8;

    FragColor = vec4(outputColor, texColor.a);
}
"""

def compile_and_link_shader(vs_source, fs_source):
    """Compiles the vertex and fragment shaders and links them into a program."""
    try:
        vertex_shader = compileShader(vs_source, GL_VERTEX_SHADER)
        fragment_shader = compileShader(fs_source, GL_FRAGMENT_SHADER)
        shader_program = compileProgram(vertex_shader, fragment_shader)
        return shader_program
    except Exception as e:
        print(f"Shader compilation/linking failed: {e}")
        return 0

def load_opencv_image_as_texture(image_path):
    """
    Loads an image using OpenCV and converts it into an OpenGL texture ID.
    NOTE: A placeholder image is used if loading fails.
    """
    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}. Using placeholder.")
    except Exception:
        # Create a simple placeholder image (512x512 blue gradient)
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        for i in range(512):
            img[:, i, 0] = 255 - i // 2
            img[:, i, 2] = i // 2
        print("Using a generated blue gradient placeholder image.")

    # Convert BGR (OpenCV default) to RGB, and flip vertically (OpenGL convention)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_flipped = np.flipud(img_rgb)

    # Generate and bind the texture
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)

    # Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Upload the image data to the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_flipped.shape[1], img_flipped.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, img_flipped)
    glGenerateMipmap(GL_TEXTURE_2D)

    return texture_id, img_flipped.shape[1], img_flipped.shape[0]

# --- Main Application Logic (Using GLFW) ---

def run_shader_app_glfw():
    # --- Configuration ---
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    IMAGE_PATH = "sample_image.jpg" # Replace with your image path

    # --- GLFW Initialization ---
    if not glfw.init():
        print("Failed to initialize GLFW")
        return

    # Set OpenGL version hints
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    # Load image first to determine window size
    texture_id, img_width, img_height = load_opencv_image_as_texture(IMAGE_PATH)

    # Calculate window size based on image aspect ratio
    aspect_ratio = img_width / img_height
    final_width = SCREEN_WIDTH
    final_height = int(SCREEN_WIDTH / aspect_ratio)

    if final_height > SCREEN_HEIGHT:
        final_height = SCREEN_HEIGHT
        final_width = int(SCREEN_HEIGHT * aspect_ratio)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(final_width, final_height, "OpenCV Image with GLSL Shader (GLFW)", None, None)
    if not window:
        glfw.terminate()
        print("Failed to create GLFW window")
        return

    glfw.make_context_current(window)

    # Set up the viewport
    glViewport(0, 0, final_width, final_height)

    # Define the key callback function for exiting
    def key_callback(window, key, scancode, action, mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            glfw.set_window_should_close(window, True)

    glfw.set_key_callback(window, key_callback)

    # --- Shader Compilation ---
    shader_program = compile_and_link_shader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE)
    if not shader_program:
        glfw.terminate()
        return

    # --- Quad Geometry Setup (The screen where the texture is drawn) ---
    quad_vertices = np.array([
        # Positions        # Texture Coords
        -1.0,  1.0, 0.0,   0.0, 1.0,   # Top-Left
         1.0,  1.0, 0.0,   1.0, 1.0,   # Top-Right
         1.0, -1.0, 0.0,   1.0, 0.0,   # Bottom-Right
        -1.0, -1.0, 0.0,   0.0, 0.0    # Bottom-Left
    ], dtype=np.float32)

    quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    # Setup VAO/VBO/EBO (identical to the Pygame script)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    # Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    # Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(3 * quad_vertices.itemsize))
    glEnableVertexAttribArray(1)

    # --- Main Loop ---
    glUseProgram(shader_program)
    
    start_time = glfw.get_time()

    while not glfw.window_should_close(window):
        # Poll and process events (required by GLFW)
        glfw.poll_events()

        # Clear the screen
        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        # Use the compiled shader program
        glUseProgram(shader_program)

        # Pass the current time to the shader for simple animation
        current_time = glfw.get_time()
        time_uniform_location = glGetUniformLocation(shader_program, "time")
        glUniform1f(time_uniform_location, current_time - start_time)

        # Bind the texture and sampler uniform
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(glGetUniformLocation(shader_program, "ourTexture"), 0)

        # Draw the quad
        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

        # Swap buffers and display the rendered frame
        glfw.swap_buffers(window)

    # --- Cleanup ---
    glDeleteTextures([texture_id])
    glDeleteProgram(shader_program)
    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO, EBO])
    
    glfw.terminate()

if __name__ == '__main__':
    run_shader_app_glfw()
