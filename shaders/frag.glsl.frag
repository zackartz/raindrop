#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragColor;
// layout(location = 1) in vec2 fragTexCoord; // If using textures

// Output color
layout(location = 0) out vec4 outColor;

// layout(binding = 1) uniform sampler2D texSampler; // If using textures

void main() {
    // Use interpolated color
    outColor = vec4(fragColor, 1.0);
    // outColor = texture(texSampler, fragTexCoord); // If using textures
}
