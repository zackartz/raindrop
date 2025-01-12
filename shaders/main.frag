#version 450

layout(location = 0) in vec3 frag_world_position;
layout(location = 1) in vec3 frag_world_normal;
layout(location = 2) in vec2 frag_tex_coord;

struct Material {
    vec4 base_color;
    float metallic_factor;
    float roughness_factor;
    vec2 _padding;
};

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 camera_pos;
    float _padding;
    Material material;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D albedo_map;
layout(set = 0, binding = 2) uniform sampler2D metallic_roughness_map;
layout(set = 0, binding = 3) uniform sampler2D normal_map;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265359;

// PBR functions
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float DistributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / max(denom, 0.0000001);
}

float GeometrySchlickGGX(float NdotV, float roughness) {
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

void main() {
    // Sample textures
    vec4 albedo = texture(albedo_map, frag_tex_coord);
    vec2 metallic_roughness = texture(metallic_roughness_map, frag_tex_coord).bg;
    vec3 normal = normalize(2.0 * texture(normal_map, frag_tex_coord).rgb - 1.0);
    
    float metallic = metallic_roughness.x * ubo.material.metallic_factor;
    float roughness = metallic_roughness.y * ubo.material.roughness_factor;

    vec3 N = normalize(normal);
    vec3 V = normalize(ubo.camera_pos - frag_world_position);
    
    // Calculate reflectance at normal incidence
    vec3 F0 = vec3(0.04); 
    F0 = mix(F0, albedo.rgb, metallic);

    // Light parameters
    vec3 light_positions[4] = vec3[](
        vec3(5.0, 5.0, 5.0),
        vec3(-5.0, 5.0, 5.0),
        vec3(5.0, -5.0, 5.0),
        vec3(-5.0, -5.0, 5.0)
    );
    vec3 light_colors[4] = vec3[](
        vec3(23.47, 21.31, 20.79),
        vec3(23.47, 21.31, 20.79),
        vec3(23.47, 21.31, 20.79),
        vec3(23.47, 21.31, 20.79)
    );

    // Reflectance equation
    vec3 Lo = vec3(0.0);
    for(int i = 0; i < 4; ++i) {
        vec3 L = normalize(light_positions[i] - frag_world_position);
        vec3 H = normalize(V + L);
        float distance = length(light_positions[i] - frag_world_position);
        float attenuation = 1.0 / (distance * distance);
        vec3 radiance = light_colors[i] * attenuation;

        // Cook-Torrance BRDF
        float NDF = DistributionGGX(N, H, roughness);   
        float G   = GeometrySmith(N, V, L, roughness);      
        vec3 F    = fresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
           
        vec3 numerator    = NDF * G * F; 
        float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0);
        vec3 specular = numerator / max(denominator, 0.001);
        
        vec3 kS = F;
        vec3 kD = vec3(1.0) - kS;
        kD *= 1.0 - metallic;	  

        float NdotL = max(dot(N, L), 0.0);        

        Lo += (kD * albedo.rgb / PI + specular) * radiance * NdotL;
    }
   
    vec3 ambient = vec3(0.03) * albedo.rgb;
    vec3 color = ambient + Lo;
	
    // HDR tonemapping
    color = color / (color + vec3(1.0));
    // gamma correction
    color = pow(color, vec3(1.0/2.2)); 

    out_color = vec4(color, albedo.a);
}
