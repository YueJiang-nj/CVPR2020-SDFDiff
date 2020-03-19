#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <ATen/ATen.h>

#define PI 3.14159265358979323846f
namespace {
    
    __device__ __forceinline__ float DegToRad(const float &deg) { return (deg * (PI / 180.f)); }

    
    __device__ __forceinline__ float length(
                            const float x, 
                            const float y, 
                            const float z) {
        return sqrtf(powf(x, 2) + powf(y, 2) + powf(z, 2));
    }

    // Cross product
    __device__ __forceinline__ float cross_x(
                            const float a_x, 
                            const float a_y, 
                            const float a_z, 
                            const float b_x, 
                            const float b_y, 
                            const float b_z) {
        return a_y * b_z - a_z * b_y;
    }

    
    __device__ __forceinline__ float cross_y(
                            const float a_x, 
                            const float a_y, 
                            const float a_z, 
                            const float b_x, 
                            const float b_y, 
                            const float b_z) {
        return a_z * b_x - a_x * b_z;
    }

    
    __device__ __forceinline__ float cross_z(
                            const float a_x, 
                            const float a_y, 
                            const float a_z, 
                            const float b_x, 
                            const float b_y, 
                            const float b_z) {
        return a_x * b_y - a_y * b_x;
    }

    __global__ void GenerateRay(
                float* origins, 
                float* directions,
                float* origin_image_distances, 
                float* pixel_distances,
                const int width, 
                const int height, 
                const float eye_x, 
                const float eye_y, 
                const float eye_z) {

        const float at_x = 0;
        const float at_y = 0;
        const float at_z = 0;
        const float up_x = 0;
        const float up_y = 1;
        const float up_z = 0;

        // Compute camera view volume
        const float top = tan(DegToRad(30));
        const float bottom = -top;
        const float right = (__int2float_rd(width) / __int2float_rd(height)) * top;
        const float left = -right;
        
        // Compute local base
        const float w_x = (eye_x - at_x) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
        const float w_y = (eye_y - at_y) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
        const float w_z = (eye_z - at_z) / length(eye_x - at_x, eye_y - at_y, eye_z - at_z);
        const float cross_up_w_x = cross_x(up_x, up_y, up_z, w_x, w_y, w_z);
        const float cross_up_w_y = cross_y(up_x, up_y, up_z, w_x, w_y, w_z);
        const float cross_up_w_z = cross_z(up_x, up_y, up_z, w_x, w_y, w_z);
        const float u_x = (cross_up_w_x) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
        const float u_y = (cross_up_w_y) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
        const float u_z = (cross_up_w_z) / length(cross_up_w_x, cross_up_w_y, cross_up_w_z);
        const float v_x = cross_x(w_x, w_y, w_z, u_x, u_y, u_z);
        const float v_y = cross_y(w_x, w_y, w_z, u_x, u_y, u_z);
        const float v_z = cross_z(w_x, w_y, w_z, u_x, u_y, u_z);


        const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (pixel_index < width * height) {
            const int x = pixel_index % width;
            const int y = pixel_index / width;
            const int i = 3 * pixel_index;

            // Compute point on view plane
            // Ray passes through the center of the pixel
            const float view_plane_x = left + (right - left) * (__int2float_rd(x) + 0.5) / __int2float_rd(width);
            const float view_plane_y = top - (top - bottom) * (__int2float_rd(y) + 0.5) / __int2float_rd(height);
            const float s_x = view_plane_x * u_x + view_plane_y * v_x - w_x;
            const float s_y = view_plane_x * u_y + view_plane_y * v_y - w_y;
            const float s_z = view_plane_x * u_z + view_plane_y * v_z - w_z;
            origins[i] = eye_x;
            origins[i+1] = eye_y;
            origins[i+2] = eye_z;

            
            directions[i] = s_x / length(s_x, s_y, s_z);
            directions[i+1] = s_y / length(s_x, s_y, s_z);
            directions[i+2] = s_z / length(s_x, s_y, s_z);

            origin_image_distances[pixel_index] = length(s_x, s_y, s_z);
            pixel_distances[pixel_index] = (right - left) / __int2float_rd(width);
            
        }
    }  

    // Check if a point is inside
    __device__ __forceinline__ bool InsideBoundingBox(
                                        const float p_x, 
                                        const float p_y, 
                                        const float p_z,
                                        const float bounding_box_min_x,
                                        const float bounding_box_min_y,
                                        const float bounding_box_min_z,
                                        const float bounding_box_max_x,
                                        const float bounding_box_max_y,
                                        const float bounding_box_max_z) {

        return (p_x >= bounding_box_min_x) && (p_x <= bounding_box_max_x) &&
               (p_y >= bounding_box_min_y) && (p_y <= bounding_box_max_y) &&
               (p_z >= bounding_box_min_z) && (p_z <= bounding_box_max_z);
    }

    // Compute the distance along the ray between the point and the bounding box  
    __device__ float Distance(
        const float reached_point_x,
        const float reached_point_y,
        const float reached_point_z,
        float direction_x,
        float direction_y,
        float direction_z,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z) {

        float dist = -1.f;
        direction_x = direction_x / length(direction_x, direction_y, direction_z);
        direction_y = direction_y / length(direction_x, direction_y, direction_z);
        direction_z = direction_z / length(direction_x, direction_y, direction_z);

        // For each axis count any excess distance outside box extents
        float v = reached_point_x;
        float d = direction_x;
        if (dist == -1) {
            if ((v < bounding_box_min_x) && (d > 0)) { dist = (bounding_box_min_x - v) / d; }
            if ((v > bounding_box_max_x) && (d < 0)) { dist = (bounding_box_max_x - v) / d; }
        } else {
            if ((v < bounding_box_min_x) && (d > 0)) { dist = fmaxf(dist, (bounding_box_min_x - v) / d); }
            if ((v > bounding_box_max_x) && (d < 0)) { dist = fmaxf(dist, (bounding_box_max_x - v) / d); }
        }

        v = reached_point_y;
        d = direction_y;
        if (dist == -1) {
            if ((v < bounding_box_min_y) && (d > 0)) { dist = (bounding_box_min_y - v) / d; }
            if ((v > bounding_box_max_y) && (d < 0)) { dist = (bounding_box_max_y - v) / d; }
        } else {
            if ((v < bounding_box_min_y) && (d > 0)) { dist = fmaxf(dist, (bounding_box_min_y - v) / d); }
            if ((v > bounding_box_max_y) && (d < 0)) { dist = fmaxf(dist, (bounding_box_max_y - v) / d); }
        }

        v = reached_point_z;
        d = direction_z;
        if (dist == -1) {
            if ((v < bounding_box_min_z) && (d > 0)) { dist = (bounding_box_min_z - v) / d; }
            if ((v > bounding_box_max_z) && (d < 0)) { dist = (bounding_box_max_z - v) / d; }
        } else {
            if ((v < bounding_box_min_z) && (d > 0)) { dist = fmaxf(dist, (bounding_box_min_z - v) / d); }
            if ((v > bounding_box_max_z) && (d < 0)) { dist = fmaxf(dist, (bounding_box_max_z - v) / d); }
        }

        return dist;
    }

    __device__ __forceinline__ int flat(float const x, float const y, float const z,
                                        int const grid_res_x, int const grid_res_y, int const grid_res_z) {
        return __int2float_rd(z) + __int2float_rd(y) * grid_res_z + __int2float_rd(x) * grid_res_z * grid_res_y;
    }

    // Get the signed distance value at the specific point
    __device__ float ValueAt(
        const float* grid,
        const float reached_point_x,
        const float reached_point_y,
        const float reached_point_z, 
        const float direction_x,
        const float direction_y,
        const float direction_z,
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x, 
        const int grid_res_y,
        const int grid_res_z,
        const bool first_time) {
        
        // Check if we are outside the BBOX
        if (!InsideBoundingBox(reached_point_x, reached_point_y, reached_point_z, 
                               bounding_box_min_x,
                               bounding_box_min_y,
                               bounding_box_min_z,
                               bounding_box_max_x,
                               bounding_box_max_y,
                               bounding_box_max_z)) {

            // If it is the first time, then the ray has not entered the grid
            if (first_time) {
                 
                return Distance(reached_point_x, reached_point_y, reached_point_z,
                                direction_x, direction_y, direction_z,
                                bounding_box_min_x,
                                bounding_box_min_y,
                                bounding_box_min_z,
                                bounding_box_max_x,
                                bounding_box_max_y,
                                bounding_box_max_z) + 0.00001f;
            }

            // Otherwise, the ray has left the grid
            else {
                return -1;
            }
        }
        
        // Compute voxel size
        float voxel_size = (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1);

        // Compute the the minimum point of the intersecting voxel
        float min_index_x = floorf((reached_point_x - bounding_box_min_x) / voxel_size);
        float min_index_y = floorf((reached_point_y - bounding_box_min_y) / voxel_size);
        float min_index_z = floorf((reached_point_z - bounding_box_min_z) / voxel_size);

        // Check whether the ray intersects the vertex with the last index of the axis
        // If so, we should record the previous index
        if (min_index_x == (bounding_box_max_x - bounding_box_min_x) / voxel_size) {
            min_index_x = (bounding_box_max_x - bounding_box_min_x) / voxel_size - 1;
        }
        if (min_index_y == (bounding_box_max_y - bounding_box_min_y) / voxel_size) {
            min_index_y = (bounding_box_max_y - bounding_box_min_y) / voxel_size - 1;
        }
        if (min_index_z == (bounding_box_max_z - bounding_box_min_z) / voxel_size) {
            min_index_z = (bounding_box_max_z - bounding_box_min_z) / voxel_size - 1;
        }

        // Linear interpolate along x axis the eight values
        const float tx = (reached_point_x - (bounding_box_min_x + min_index_x * voxel_size)) / voxel_size;
        const float c01 = (1.f - tx) * grid[flat(min_index_x, min_index_y, min_index_z, grid_res_x, grid_res_y, grid_res_z)]
         + tx * grid[flat(min_index_x+1, min_index_y, min_index_z, grid_res_x, grid_res_y, grid_res_z)];
        const float c23 = (1.f - tx) * grid[flat(min_index_x, min_index_y+1, min_index_z, grid_res_x, grid_res_y, grid_res_z)]
         + tx * grid[flat(min_index_x+1, min_index_y+1, min_index_z, grid_res_x, grid_res_y, grid_res_z)];
        const float c45 = (1.f - tx) * grid[flat(min_index_x, min_index_y, min_index_z+1, grid_res_x, grid_res_y, grid_res_z)]
         + tx * grid[flat(min_index_x+1, min_index_y, min_index_z+1, grid_res_x, grid_res_y, grid_res_z)];
        const float c67 = (1.f - tx) * grid[flat(min_index_x, min_index_y+1, min_index_z+1, grid_res_x, grid_res_y, grid_res_z)]
         + tx * grid[flat(min_index_x+1, min_index_y+1, min_index_z+1, grid_res_x, grid_res_y, grid_res_z)];
       
        // Linear interpolate along the y axis
        const float ty = (reached_point_y - (bounding_box_min_y + min_index_y * voxel_size)) / voxel_size;
        const float c0 = (1.f - ty) * c01 + ty * c23;
        const float c1 = (1.f - ty) * c45 + ty * c67;

        // Return final value interpolated along z
        const float tz = (reached_point_z - (bounding_box_min_z + min_index_z * voxel_size)) / voxel_size;      

        return (1.f - tz) * c0 + tz * c1;
    }

    // Compute the intersection of the ray and the grid
    // The intersection procedure uses ray marching to check if we have an interaction with the stored surface    
    __global__ void Intersect(
        const float* grid,
        const float* origins,
        const float* directions,
        const float* origin_image_distances, 
        const float* pixel_distances, 
        const float bounding_box_min_x,
        const float bounding_box_min_y,
        const float bounding_box_min_z,
        const float bounding_box_max_x,
        const float bounding_box_max_y,
        const float bounding_box_max_z,
        const int grid_res_x, 
        const int grid_res_y,
        const int grid_res_z,
        float* voxel_position,
        float* intersection_pos, 
        const int width, 
        const int height) {
        
        // Compute voxel size
        const float voxel_size = (bounding_box_max_x - bounding_box_min_x) / (grid_res_x - 1);

        // Define constant values
        const int max_steps = 1000;
        bool first_time = true;
        float depth = 0;
        int gotten_result = 0;

        const int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;

        if (pixel_index < width * height) {

            const int i = 3 * pixel_index;
            
            for (int steps = 0; steps < max_steps; steps++) {

                float reached_point_x = origins[i] + depth * directions[i];
                float reached_point_y = origins[i+1] + depth * directions[i+1];
                float reached_point_z = origins[i+2] + depth * directions[i+2];

                // Get the signed distance value for the point the ray reaches
                const float distance = ValueAt(grid, reached_point_x, reached_point_y, reached_point_z,
                                               directions[i], directions[i+1], directions[i+2],
                                               bounding_box_min_x,
                                               bounding_box_min_y,
                                               bounding_box_min_z,
                                               bounding_box_max_x,
                                               bounding_box_max_y,
                                               bounding_box_max_z,
                                               grid_res_x, 
                                               grid_res_y,
                                               grid_res_z, first_time);
                first_time = false;

                // Check if the ray is going ourside the bounding box
                if (distance == -1) {
                    voxel_position[i] = -1;
                    voxel_position[i+1] = -1;
                    voxel_position[i+2] = -1;
                    intersection_pos[i] = -1;
                    intersection_pos[i+1] = -1;
                    intersection_pos[i+2] = -1;
                    gotten_result = 1;
                    break;
                }

                // Check if we are close enough to the surface
               if (distance < pixel_distances[pixel_index] / origin_image_distances[pixel_index] * depth && distance) {

                    // Compute the the minimum point of the intersecting voxel
                    voxel_position[i] = floorf((reached_point_x - bounding_box_min_x) / voxel_size);
                    voxel_position[i+1] = floorf((reached_point_y - bounding_box_min_y) / voxel_size);
                    voxel_position[i+2] = floorf((reached_point_z - bounding_box_min_z) / voxel_size);
                    if (voxel_position[i] == grid_res_x - 1) {
                        voxel_position[i] = voxel_position[i] - 1;
                    }
                    if (voxel_position[i+1] == grid_res_x - 1) {
                        voxel_position[i+1] = voxel_position[i+1] - 1;
                    }
                    if (voxel_position[i+2] == grid_res_x - 1) {
                        voxel_position[i+2] = voxel_position[i+2] - 1;
                    }
                    intersection_pos[i] = reached_point_x;
                    intersection_pos[i+1] = reached_point_y;
                    intersection_pos[i+2] = reached_point_z;
                    gotten_result = 1;
                    break;
                }

                // Increase distance
                depth += distance;

            }
            
            if (gotten_result == 0) {

                // No intersections
                voxel_position[i] = -1;
                voxel_position[i+1] = -1;
                voxel_position[i+2] = -1;
                intersection_pos[i] = -1;
                intersection_pos[i+1] = -1;
                intersection_pos[i+2] = -1;
            }
        }
    }
} // namespace

// Ray marching to get the first corner position of the voxel the ray intersects
std::vector<at::Tensor> ray_matching_cuda(
                           const at::Tensor w_h_3,
                           const at::Tensor w_h,
                           const at::Tensor grid, 
                           const int width, 
                           const int height,
                           const float bounding_box_min_x,
                           const float bounding_box_min_y,
                           const float bounding_box_min_z,
                           const float bounding_box_max_x,
                           const float bounding_box_max_y,
                           const float bounding_box_max_z,
                           const int grid_res_x, 
                           const int grid_res_y,
                           const int grid_res_z, 
                           const float eye_x,  
                           const float eye_y,  
                           const float eye_z) {

    const int thread = 512;

    at::Tensor origins = at::zeros_like(w_h_3);
    at::Tensor directions = at::zeros_like(w_h_3);
    at::Tensor origin_image_distances = at::zeros_like(w_h);
    at::Tensor pixel_distances = at::zeros_like(w_h);

        GenerateRay<<<(width * height + thread - 1) / thread, thread>>>(
                                     origins.data<float>(), 
                                     directions.data<float>(), 
                                     origin_image_distances.data<float>(),
                                     pixel_distances.data<float>(), 
                                     width, 
                                     height,
                                     eye_x, 
                                     eye_y, 
                                     eye_z);

    at::Tensor voxel_position = at::zeros_like(w_h_3);
    at::Tensor intersection_pos = at::zeros_like(w_h_3);

    Intersect<<<(width * height + thread - 1) / thread, thread>>>(
                                                    grid.data<float>(), 
                                                    origins.data<float>(), 
                                                    directions.data<float>(),
                                                    origin_image_distances.data<float>(),
                                                    pixel_distances.data<float>(), 
                                                    bounding_box_min_x,
                                                    bounding_box_min_y,
                                                    bounding_box_min_z,
                                                    bounding_box_max_x,
                                                    bounding_box_max_y,
                                                    bounding_box_max_z,
                                                    grid_res_x, 
                                                    grid_res_y,
                                                    grid_res_z, 
                                                    voxel_position.data<float>(), 
                                                    intersection_pos.data<float>(), 
                                                    width, 
                                                    height);        

    return {intersection_pos, voxel_position, directions};
}



