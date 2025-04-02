#version 450

// Matches Vertex struct attribute descriptions
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
// layout(location = 2) in vec2 inTexCoord; // If you add texture coords

// Matches UniformBufferObject struct and descriptor set layout binding
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Output to fragment shader
layout(location = 0) out vec3 fragColor;
// layout(location = 1) out vec2 fragTexCoord; // If you add texture coords

void main() {
    // Transform position: Model -> World -> View -> Clip space
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    // Pass color (and other attributes) through
    fragColor = inColor;
    // fragTexCoord = inTexCoord;
}
