
#version 450

// Input from vertex shader
layout(location = 0) in vec3 fragNormal; // Receive normal
layout(location = 1) in vec2 fragTexCoord; // Receive texture coordinates

// Output color
layout(location = 0) out vec4 outColor;

// Descriptor set for material properties (Set 1)
layout(set = 1, binding = 0) uniform sampler2D baseColorSampler;

// Optional: Pass material factors via another UBO or Push Constants if needed
// layout(set = 1, binding = 1) uniform MaterialFactors {
//     vec4 baseColorFactor;
// } materialFactors;

void main() {
    // Sample the texture
    vec4 texColor = texture(baseColorSampler, fragTexCoord);

    // Use the texture color
    // You might multiply by baseColorFactor here if you pass it
    // outColor = texColor * materialFactors.baseColorFactor;
    outColor = texColor;

    // Basic fallback if texture alpha is zero (or use baseColorFactor)
    if (outColor.a == 0.0) {
        outColor = vec4(0.8, 0.8, 0.8, 1.0); // Default grey
    }

    // You could add basic lighting using fragNormal here later
}
