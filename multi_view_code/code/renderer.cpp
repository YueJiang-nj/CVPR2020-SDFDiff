#include <torch/torch.h>
#include <vector>

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
                               const float eye_z);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> ray_matching(
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
    CHECK_INPUT(w_h_3);
    CHECK_INPUT(w_h);
    CHECK_INPUT(grid);

    return ray_matching_cuda(w_h_3, w_h, grid, width, height, 
                                bounding_box_min_x,
                                bounding_box_min_y,
                                bounding_box_min_z,
                                bounding_box_max_x,
                                bounding_box_max_y,
                                bounding_box_max_z,
                                grid_res_x, 
                                grid_res_y,
                                grid_res_z, 
                                eye_x,  eye_y,  eye_z);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_matching", &ray_matching, "Ray Matching");
}