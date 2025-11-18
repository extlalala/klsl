# klsl
A Kotlin-based Shader Language Emulator, syntax like glsl
KLSL is a lightweight, Kotlin-implemented shader language emulator designed for generating procedural graphics and visual effects. It provides a domain-specific language (DSL) syntax similar to GLSL (OpenGL Shading Language), allowing developers to write shader-like code in Kotlin for image generation and processing tasks.

Key Features
1. GLSL-like Syntax in Kotlin
Familiar vector/matrix operations (vec2, vec3, vec4, mat2)
Common shader functions (length(), dot(), step(), smoothstep(), etc.)
Mathematical operations (trigonometric, exponential, etc.)
Texture-like coordinate operations (though without actual texture sampling)
2. Multi-threaded Rendering Backend
Efficient PPM (Portable Pixmap Format) output backend
Automatic parallel processing using all available CPU cores
Optimized for both small and large image sizes
3. Flexible Frontend Interface
Uniform variable system for passing parameters to shaders
Time-based animation support through iTime uniform
Resolution-aware rendering through iResolution uniform
4. Example Shaders
The package includes several example shaders demonstrating various effects:

Swirling color patterns
Grid-based animations
Time-varying geometric transformations
Procedural texture generation
Usage Example
```kotlin
fun main() {
    val path = "output.ppm"
    val out = Files.newOutputStream(Path.of(path), StandardOpenOption.WRITE).buffered()
    
    out.use { 
        val shader = klsl(DefaultFrontend(PPMBackend(it))) {
            mainImage {
                val U = (C - 0.5f * iResolution.xy) / iResolution.y * 5.0f
                val p = vec3(atan(U.y, U.x) * 3.0f, length(U) * 2.0f, iTime * 0.5f)
                val h = sin(p.x + cos(p.y) * 2.0f) * 0.5f + 0.5f
                O = vec4(h * vec3(1f, 0.3f, sin(iTime) * 0.5f + 0.5f), 1f) * 
                    (1.0f - smoothstep(0.4f, 0.41f, abs(mod(p.y, 1.0f) - 0.5f)))
            }
        }
        shader.render(ivec2(400, 400), 0f)
    }
}
```
Implementation Details
Core Components
Vector/Matrix Classes: vec2, vec3, vec4, and mat2 with operator overloading
Frontend Interface: Defines the shader execution environment
Backend System: Handles actual pixel generation (currently only PPM format)
Shader DSL: Kotlin DSL for writing shader-like code
Performance Considerations
The PPMBackend automatically parallelizes rendering across all available CPU cores
For small images, it falls back to single-threaded processing for better efficiency
Memory-efficient byte array handling for pixel data
Limitations
No actual texture sampling support (only procedural generation)
Limited to PPM output format (though extensible to other formats)
Some GLSL functions are simplified or stubbed (like dFdx/dFdy)
No hardware acceleration - purely CPU-based
Potential Use Cases
Procedural texture generation
Algorithm visualization
Educational purposes (learning shader concepts)
Rapid prototyping of visual effects
Generative art experiments
Getting Started
Clone the repository or copy the source files
Add to your Kotlin project
Implement your own shaders using the provided DSL
Render to PPM files or extend with custom backends
Extensions Possible
Add more output formats (PNG, JPEG, etc.)
Implement additional GLSL functions
Add noise functions (Perlin, Simplex, etc.)
Create a GUI frontend for interactive shader development
Add support for uniform texture inputs (though would require image loading)
KLSL provides a fun and educational way to explore shader programming concepts without needing OpenGL/WebGL knowledge, right from your Kotlin environment.
