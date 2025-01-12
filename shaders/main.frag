#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_world_normal;
layout(location = 2) in vec2 frag_tex_coord;

layout(set = 0, binding = 1) uniform sampler2D tex_sampler;

layout(location = 0) out vec4 out_color;

void main() {
    // Sample the texture
    vec4 base_color = texture(tex_sampler, frag_tex_coord);
    
    // Output final color
    out_color = vec4(base_color.rgb, base_color.a);
}
