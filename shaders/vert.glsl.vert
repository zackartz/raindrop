#version 450

// INPUTS from Vertex Buffer (matching Vertex struct)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord; // <<< MUST be vec2

// UNIFORMS (Set 0)
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// PUSH CONSTANTS
layout(push_constant) uniform PushConstants {
    mat4 model;
} pushConstants;

// OUTPUTS to Fragment Shader
layout(location = 0) out vec3 fragNormal; // Location 0 for Normal
layout(location = 1) out vec2 fragTexCoord; // Location 1 for TexCoord

void main() {
    vec4 worldPos = pushConstants.model * vec4(inPosition, 1.0);

    // Calculate final position
    gl_Position = ubo.proj * ubo.view * worldPos;

    // --- Pass attributes to Fragment Shader ---

    // Pass world-space normal (adjust calculation if needed)
    // Ensure fragNormal is assigned a vec3
    fragNormal = normalize(mat3(transpose(inverse(pushConstants.model))) * inNormal);

    // Pass texture coordinates (ensure inTexCoord is vec2)
    // Ensure fragTexCoord is assigned a vec2
    fragTexCoord = inTexCoord;
}
