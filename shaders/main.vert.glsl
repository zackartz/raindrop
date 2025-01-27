#version 450

// Vertex inputs
layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_tex_coord;

// Uniform buffer
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

// Vertex outputs
layout(location = 0) out vec3 out_world_position;
layout(location = 1) out vec3 out_world_normal;
layout(location = 2) out vec2 out_tex_coord;

void main() {
    // Transform position to world space
    vec4 pos = ubo.model * vec4(in_pos, 1.0);
    out_world_position = (pos / pos.w).xyz;

    // Transform normal to world space
    mat3 normal_matrix = transpose(inverse(mat3(ubo.model)));
    out_world_normal = normal_matrix * in_normal;

    // Calculate clip space position
    gl_Position = ubo.proj * ubo.view * pos;
    out_tex_coord = in_tex_coord;
}
