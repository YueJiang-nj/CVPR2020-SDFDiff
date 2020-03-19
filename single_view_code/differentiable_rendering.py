from __future__ import print_function
import torch
import math
import torchvision
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import renderer
import time
import sys

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def read_txt(file_path, grid_res_x, grid_res_y, grid_res_z):
    with open(file_path) as file:
        grid = Tensor(grid_res_x, grid_res_y, grid_res_z)
        for i in range(grid_res_x):
            for j in range(grid_res_y):
                for k in range(grid_res_z):
                    grid[i][j][k] = float(file.readline())
    print (grid)
    
    return grid

# Read a file and create a sdf grid with target_grid_res
def read_sdf(file_path, target_grid_res, target_bounding_box_min, target_bounding_box_max, target_voxel_size):

    with open(file_path) as file:  
        line = file.readline()

        # Get grid resolutions
        grid_res = line.split()
        grid_res_x = int(grid_res[0])
        grid_res_y = int(grid_res[1])
        grid_res_z = int(grid_res[2])

        # Get bounding box min
        line = file.readline()
        bounding_box_min = line.split()
        bounding_box_min_x = float(bounding_box_min[0])
        bounding_box_min_y = float(bounding_box_min[1])
        bounding_box_min_z = float(bounding_box_min[2])

        line = file.readline()
        voxel_size = float(line)

        # max bounding box (we need to plus 0.0001 to avoid round error)
        bounding_box_max_x = bounding_box_min_x + voxel_size * (grid_res_x - 1)# + 0.0001
        bounding_box_max_y = bounding_box_min_y + voxel_size * (grid_res_y - 1) #+ 0.0001
        bounding_box_max_z = bounding_box_min_z + voxel_size * (grid_res_z - 1) #+ 0.0001

        min_bounding_box_min = min(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z) 
        print(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z)
        max_bounding_box_max = max(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z) 
        print(bounding_box_max_x, bounding_box_max_y, bounding_box_max_z)
        max_dist = max(bounding_box_max_x - bounding_box_min_x, bounding_box_max_y - bounding_box_min_y, bounding_box_max_z - bounding_box_min_z)

        max_grid_res = max(grid_res_x, grid_res_y, grid_res_z)

        grid = []
        for i in range(grid_res_x):
            grid.append([])
            for j in range(grid_res_y):
                grid[i].append([])
                for k in range(grid_res_z):
                    grid[i][j].append(2)

        for i in range(grid_res_z):
            for j in range(grid_res_y):
                for k in range(grid_res_x):
                    grid_value = float(file.readline())
                    grid[k][j][i] = grid_value

        grid = Tensor(grid)

        target_grid = Tensor(target_grid_res, target_grid_res, target_grid_res)

        linear_space_x = torch.linspace(0, target_grid_res-1, target_grid_res)
        linear_space_y = torch.linspace(0, target_grid_res-1, target_grid_res)
        linear_space_z = torch.linspace(0, target_grid_res-1, target_grid_res)
        first_loop = linear_space_x.repeat(target_grid_res * target_grid_res, 1).t().contiguous().view(-1).unsqueeze_(1)
        second_loop = linear_space_y.repeat(target_grid_res, target_grid_res).t().contiguous().view(-1).unsqueeze_(1)
        third_loop = linear_space_z.repeat(target_grid_res * target_grid_res).unsqueeze_(1)
        loop = torch.cat((first_loop, second_loop, third_loop), 1).cuda()

        min_x = Tensor([bounding_box_min_x]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        min_y = Tensor([bounding_box_min_y]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        min_z = Tensor([bounding_box_min_z]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        bounding_min_matrix = torch.cat((min_x, min_y, min_z), 1)

        move_to_center_x = Tensor([(max_dist - (bounding_box_max_x - bounding_box_min_x)) / 2]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        move_to_center_y = Tensor([(max_dist - (bounding_box_max_y - bounding_box_min_y)) / 2]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        move_to_center_z = Tensor([(max_dist - (bounding_box_max_z - bounding_box_min_z)) / 2]).repeat(target_grid_res*target_grid_res*target_grid_res, 1)
        move_to_center_matrix = torch.cat((move_to_center_x, move_to_center_y, move_to_center_z), 1)
        
        # Get the position of the grid points in the refined grid
        points = bounding_min_matrix + target_voxel_size * max_dist / (target_bounding_box_max - target_bounding_box_min) * loop - move_to_center_matrix
        if points[(points[:, 0] < bounding_box_min_x)].shape[0] != 0:
            points[(points[:, 0] < bounding_box_min_x)] = Tensor([bounding_box_max_x, bounding_box_max_y, bounding_box_max_z]).view(1,3)
        if points[(points[:, 1] < bounding_box_min_y)].shape[0] != 0:
            points[(points[:, 1] < bounding_box_min_y)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 2] < bounding_box_min_z)].shape[0] != 0:
            points[(points[:, 2] < bounding_box_min_z)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 0] > bounding_box_max_x)].shape[0] != 0:
            points[(points[:, 0] > bounding_box_max_x)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 1] > bounding_box_max_y)].shape[0] != 0:
            points[(points[:, 1] > bounding_box_max_y)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        if points[(points[:, 2] > bounding_box_max_z)].shape[0] != 0:
            points[(points[:, 2] > bounding_box_max_z)] = Tensor([bounding_box_max_x, bounding_box_min_y, bounding_box_min_z]).view(1,3)
        voxel_min_point_index_x = torch.floor((points[:,0].unsqueeze_(1) - min_x) / voxel_size).clamp(max=grid_res_x-2)
        voxel_min_point_index_y = torch.floor((points[:,1].unsqueeze_(1) - min_y) / voxel_size).clamp(max=grid_res_y-2)
        voxel_min_point_index_z = torch.floor((points[:,2].unsqueeze_(1) - min_z) / voxel_size).clamp(max=grid_res_z-2)
        voxel_min_point_index = torch.cat((voxel_min_point_index_x, voxel_min_point_index_y, voxel_min_point_index_z), 1)
        voxel_min_point = bounding_min_matrix + voxel_min_point_index * voxel_size

        # Compute the sdf value of the grid points in the refined grid
        target_grid = calculate_sdf_value(grid, points, voxel_min_point, voxel_min_point_index, voxel_size, grid_res_x, grid_res_y, grid_res_z).view(target_grid_res, target_grid_res, target_grid_res)
        return target_grid

        
def grid_construction_cube(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a cube with size 2
    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    cube_left_bound_index = float(grid_res - 1) / 4;
    cube_right_bound_index = float(grid_res - 1) / 4 * 3;
    cube_center = float(grid_res - 1) / 2;

    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                if (i >= cube_left_bound_index and i <= cube_right_bound_index and
                    j >= cube_left_bound_index and j <= cube_right_bound_index and
                    k >= cube_left_bound_index and k <= cube_right_bound_index):
                    grid[i,j,k] = voxel_size * max(abs(i - cube_center), abs(j - cube_center), abs(k - cube_center)) - 1;
                else:
                    grid[i,j,k] = math.sqrt(pow(voxel_size * (max(i - cube_right_bound_index, cube_left_bound_index - i, 0)), 2) +
                                    pow(voxel_size * (max(j - cube_right_bound_index, cube_left_bound_index - j, 0)), 2) +
                                    pow(voxel_size * (max(k - cube_right_bound_index, cube_left_bound_index - k, 0)), 2));
    return grid

def grid_construction_torus(grid_res, bounding_box_min, bounding_box_max):
    
    # radius of the circle between the two circles
    radius_big = 1.5

    # radius of the small circle
    radius_small = 0.5

    voxel_size = (bounding_box_max - bounding_box_min) / (grid_res - 1)
    grid = Tensor(grid_res, grid_res, grid_res)
    for i in range(grid_res):
        for j in range(grid_res):
            for k in range(grid_res):
                x = bounding_box_min + voxel_size * i
                y = bounding_box_min + voxel_size * j
                z = bounding_box_min + voxel_size * k

                grid[i,j,k] = math.sqrt(math.pow((math.sqrt(math.pow(y, 2) + math.pow(z, 2)) - radius_big), 2)
                              + math.pow(x, 2)) - radius_small;

    return grid



def grid_construction_sphere_big(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a sphere with radius 1
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1.6
    if cuda:
        return grid.cuda()
    else:
        return grid

def grid_construction_sphere_small(grid_res, bounding_box_min, bounding_box_max):

    # Construct the sdf grid for a sphere with radius 1
    linear_space = torch.linspace(bounding_box_min, bounding_box_max, grid_res)
    x_dim = linear_space.view(-1, 1).repeat(grid_res, 1, grid_res)
    y_dim = linear_space.view(1, -1).repeat(grid_res, grid_res, 1)
    z_dim = linear_space.view(-1, 1, 1).repeat(1, grid_res, grid_res)
    grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 1
    if cuda:
        return grid.cuda()
    else:
        return grid


def get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z):
    
    # largest index
    n_x = grid_res_x - 1
    n_y = grid_res_y - 1
    n_z = grid_res_z - 1

    # x-axis normal vectors
    X_1 = torch.cat((grid[1:,:,:], (3 * grid[n_x,:,:] - 3 * grid[n_x-1,:,:] + grid[n_x-2,:,:]).unsqueeze_(0)), 0)
    X_2 = torch.cat(((-3 * grid[1,:,:] + 3 * grid[0,:,:] + grid[2,:,:]).unsqueeze_(0), grid[:n_x,:,:]), 0)
    grid_normal_x = (X_1 - X_2) / (2 * voxel_size)

    # y-axis normal vectors
    Y_1 = torch.cat((grid[:,1:,:], (3 * grid[:,n_y,:] - 3 * grid[:,n_y-1,:] + grid[:,n_y-2,:]).unsqueeze_(1)), 1)
    Y_2 = torch.cat(((-3 * grid[:,1,:] + 3 * grid[:,0,:] + grid[:,2,:]).unsqueeze_(1), grid[:,:n_y,:]), 1)
    grid_normal_y = (Y_1 - Y_2) / (2 * voxel_size)

    # z-axis normal vectors
    Z_1 = torch.cat((grid[:,:,1:], (3 * grid[:,:,n_z] - 3 * grid[:,:,n_z-1] + grid[:,:,n_z-2]).unsqueeze_(2)), 2)
    Z_2 = torch.cat(((-3 * grid[:,:,1] + 3 * grid[:,:,0] + grid[:,:,2]).unsqueeze_(2), grid[:,:,:n_z]), 2)
    grid_normal_z = (Z_1 - Z_2) / (2 * voxel_size)


    return [grid_normal_x, grid_normal_y, grid_normal_z]


def get_intersection_normal(intersection_grid_normal, intersection_pos, voxel_min_point, voxel_size):

    # Compute parameters
    tx = (intersection_pos[:,:,0] - voxel_min_point[:,:,0]) / voxel_size
    ty = (intersection_pos[:,:,1] - voxel_min_point[:,:,1]) / voxel_size
    tz = (intersection_pos[:,:,2] - voxel_min_point[:,:,2]) / voxel_size

    intersection_normal = (1 - tz) * (1 - ty) * (1 - tx) * intersection_grid_normal[:,:,0] \
                        + tz * (1 - ty) * (1 - tx) * intersection_grid_normal[:,:,1] \
                        + (1 - tz) * ty * (1 - tx) * intersection_grid_normal[:,:,2] \
                        + tz * ty * (1 - tx) * intersection_grid_normal[:,:,3] \
                        + (1 - tz) * (1 - ty) * tx * intersection_grid_normal[:,:,4] \
                        + tz * (1 - ty) * tx * intersection_grid_normal[:,:,5] \
                        + (1 - tz) * ty * tx * intersection_grid_normal[:,:,6] \
                        + tz * ty * tx * intersection_grid_normal[:,:,7]

    return intersection_normal


# Do one more step for ray matching
def calculate_sdf_value(grid, points, voxel_min_point, voxel_min_point_index, voxel_size, grid_res_x, grid_res_y, grid_res_z):

    string = ""
    # Linear interpolate along x axis the eight values
    tx = (points[:,0] - voxel_min_point[:,0]) / voxel_size;
    string = string + "\n\nvoxel_size: \n" + str(voxel_size)
    string = string + "\n\ntx: \n" + str(tx)
    # print(grid.shape)

    if cuda:
        tx = tx.cuda()
        x = voxel_min_point_index.long()[:,0]
        y = voxel_min_point_index.long()[:,1]
        z = voxel_min_point_index.long()[:,2]

        string = string + "\n\nx: \n" + str(x)
        string = string + "\n\ny: \n" + str(y)
        string = string + "\n\nz: \n" + str(z)

        c01 = (1 - tx) * grid[x,y,z] + tx * grid[x+1,y,z];
        c23 = (1 - tx) * grid[x,y+1,z] + tx * grid[x+1,y+1,z];
        c45 = (1 - tx) * grid[x,y,z+1] + tx * grid[x+1,y,z+1];
        c67 = (1 - tx) * grid[x,y+1,z+1] + tx * grid[x+1,y+1,z+1];

        string = string + "\n\n(1 - tx): \n" + str((1 - tx))
        string = string + "\n\ngrid[x,y,z]: \n" + str(grid[x,y,z])
        string = string + "\n\ngrid[x+1,y,z]: \n" + str(grid[x+1,y,z])
        string = string + "\n\nc01: \n" + str(c01)
        string = string + "\n\nc23: \n" + str(c23)
        string = string + "\n\nc45: \n" + str(c45)
        string = string + "\n\nc67: \n" + str(c67)

        # Linear interpolate along the y axis
        ty = (points[:,1] - voxel_min_point[:,1]) / voxel_size;
        ty = ty.cuda()
        c0 = (1 - ty) * c01 + ty * c23;
        c1 = (1 - ty) * c45 + ty * c67;

        string = string + "\n\nty: \n" + str(ty)

        string = string + "\n\nc0: \n" + str(c0)
        string = string + "\n\nc1: \n" + str(c1)

        # Return final value interpolated along z
        tz = (points[:,2] - voxel_min_point[:,2]) / voxel_size;
        tz = tz.cuda()
        string = string + "\n\ntz: \n" + str(tz)
        
    else:
        x = voxel_min_point_index.numpy()[:,0]
        y = voxel_min_point_index.numpy()[:,1]
        z = voxel_min_point_index.numpy()[:,2]

        c01 = (1 - tx) * grid[x,y,z] + tx * grid[x+1,y,z];
        c23 = (1 - tx) * grid[x,y+1,z] + tx * grid[x+1,y+1,z];
        c45 = (1 - tx) * grid[x,y,z+1] + tx * grid[x+1,y,z+1];
        c67 = (1 - tx) * grid[x,y+1,z+1] + tx * grid[x+1,y+1,z+1];

        # Linear interpolate along the y axis
        ty = (points[:,1] - voxel_min_point[:,1]) / voxel_size;
        c0 = (1 - ty) * c01 + ty * c23;
        c1 = (1 - ty) * c45 + ty * c67;

        # Return final value interpolated along z
        tz = (points[:,2] - voxel_min_point[:,2]) / voxel_size;

    result = (1 - tz) * c0 + tz * c1;

    return result


def compute_intersection_pos(grid, intersection_pos_rough, voxel_min_point, voxel_min_point_index, ray_direction, voxel_size, mask):
    
    # Linear interpolate along x axis the eight values
    tx = (intersection_pos_rough[:,:,0] - voxel_min_point[:,:,0]) / voxel_size;

    if cuda:

        x = voxel_min_point_index.long()[:,:,0]
        y = voxel_min_point_index.long()[:,:,1]
        z = voxel_min_point_index.long()[:,:,2]

        c01 = (1 - tx) * grid[x,y,z].cuda() + tx * grid[x+1,y,z].cuda();
        c23 = (1 - tx) * grid[x,y+1,z].cuda() + tx * grid[x+1,y+1,z].cuda();
        c45 = (1 - tx) * grid[x,y,z+1].cuda() + tx * grid[x+1,y,z+1].cuda();
        c67 = (1 - tx) * grid[x,y+1,z+1].cuda() + tx * grid[x+1,y+1,z+1].cuda();

    else:
        x = voxel_min_point_index.numpy()[:,:,0]
        y = voxel_min_point_index.numpy()[:,:,1]
        z = voxel_min_point_index.numpy()[:,:,2]

        c01 = (1 - tx) * grid[x,y,z] + tx * grid[x+1,y,z];
        c23 = (1 - tx) * grid[x,y+1,z] + tx * grid[x+1,y+1,z];
        c45 = (1 - tx) * grid[x,y,z+1] + tx * grid[x+1,y,z+1];
        c67 = (1 - tx) * grid[x,y+1,z+1] + tx * grid[x+1,y+1,z+1];     
           
    # Linear interpolate along the y axis
    ty = (intersection_pos_rough[:,:,1] - voxel_min_point[:,:,1]) / voxel_size;
    c0 = (1 - ty) * c01 + ty * c23;
    c1 = (1 - ty) * c45 + ty * c67;

    # Return final value interpolated along z
    tz = (intersection_pos_rough[:,:,2] - voxel_min_point[:,:,2]) / voxel_size;

    sdf_value = (1 - tz) * c0 + tz * c1;

    return (intersection_pos_rough + ray_direction * sdf_value.view(width,height,1).repeat(1,1,3))\
                            + (1 - mask.view(width,height,1).repeat(1,1,3))


def differentiable_rendering(grid, grid_res, image_res, camera):
    # print(grid_res, image_res, camera)
    # return grid
    global width, height
    width = image_res
    height = image_res

    return generate_image(-2, -2, -2, 2, 2, 2, \
    4./(grid_res-1), grid_res, grid_res, grid_res, image_res, image_res, grid, camera, False, []) 

def differentiable_rendering_silhouette(grid, grid_res, image_res, camera):
    # print(grid_res, image_res, camera)
    return generate_image(-2, -2, -2, 2, 2, 2, \
    4./(grid_res-1), grid_res, grid_res, grid_res, image_res, image_res, grid, camera, True)

def generate_image(bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
    voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height, grid, camera, back, camera_list):

    # Get normal vectors for points on the grid
    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z)

    # Generate rays
    e = camera
    
    w_h_3 = torch.zeros(width, height, 3).cuda()
    w_h = torch.zeros(width, height).cuda()
    eye_x = e[0]
    eye_y = e[1]
    eye_z = e[2]

    # Do ray tracing in cpp
    outputs = renderer.ray_matching(w_h_3, w_h, grid, width, height, bounding_box_min_x, bounding_box_min_y, bounding_box_min_z, \
    bounding_box_max_x, bounding_box_max_y, bounding_box_max_z, \
                            grid_res_x, grid_res_y, grid_res_z, \
                            eye_x, \
                            eye_y, \
                            eye_z
                            )

    # {intersection_pos, voxel_position, directions}
    intersection_pos_rough = outputs[0]
    voxel_min_point_index = outputs[1]
    ray_direction = outputs[2]

    # Initialize grid values and normals for intersection voxels
    intersection_grid_normal_x = Tensor(width, height, 8)
    intersection_grid_normal_y = Tensor(width, height, 8)
    intersection_grid_normal_z = Tensor(width, height, 8)
    intersection_grid = Tensor(width, height, 8)

    # Make the pixels with no intersections with rays be 0
    mask = (voxel_min_point_index[:,:,0] != -1).type(Tensor)

    # Get the indices of the minimum point of the intersecting voxels
    x = voxel_min_point_index[:,:,0].type(torch.cuda.LongTensor)
    y = voxel_min_point_index[:,:,1].type(torch.cuda.LongTensor)
    z = voxel_min_point_index[:,:,2].type(torch.cuda.LongTensor)
    x[x == -1] = 0
    y[y == -1] = 0
    z[z == -1] = 0

    # Get the x-axis of normal vectors for the 8 points of the intersecting voxel
    # This line is equivalent to grid_normal_x[x,y,z]
    x1 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x2 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x3 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x4 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    x5 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x6 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x7 = torch.index_select(grid_normal_x.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    x8 = torch.index_select(grid_normal_x.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2) + (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
    y1 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y2 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y3 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y4 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    y5 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y6 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y7 = torch.index_select(grid_normal_y.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    y8 = torch.index_select(grid_normal_y.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 2) + (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
    z1 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z2 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z3 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z4 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * x.view(-1)).view(x.shape).unsqueeze_(2)
    z5 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z6 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * y.view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z7 = torch.index_select(grid_normal_z.view(-1), 0, z.view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    z8 = torch.index_select(grid_normal_z.view(-1), 0, (z+1).view(-1) + grid_res_x * (y+1).view(-1) + grid_res_x * grid_res_x * (x+1).view(-1)).view(x.shape).unsqueeze_(2)
    intersection_grid_normal_z = torch.cat((z1, z2, z3, z4, z5, z6, z7, z8), 2) + (1 - mask.view(width, height, 1).repeat(1,1,8))

    # Change from grid coordinates to world coordinates
    voxel_min_point = Tensor([bounding_box_min_x, bounding_box_min_y, bounding_box_min_z]) + voxel_min_point_index * voxel_size

    intersection_pos = compute_intersection_pos(grid, intersection_pos_rough,\
                                                voxel_min_point, voxel_min_point_index,\
                                                ray_direction, voxel_size, mask)

    intersection_pos = intersection_pos * mask.repeat(3,1,1).permute(1,2,0)
    shading = Tensor(width, height).fill_(0)

    # Compute the normal vectors for the intersecting points
    intersection_normal_x = get_intersection_normal(intersection_grid_normal_x, intersection_pos, voxel_min_point, voxel_size)
    intersection_normal_y = get_intersection_normal(intersection_grid_normal_y, intersection_pos, voxel_min_point, voxel_size)
    intersection_normal_z = get_intersection_normal(intersection_grid_normal_z, intersection_pos, voxel_min_point, voxel_size)
    
    # Put all the xyz-axis of the normal vectors into a single matrix
    intersection_normal_x_resize = intersection_normal_x.unsqueeze_(2)
    intersection_normal_y_resize = intersection_normal_y.unsqueeze_(2)
    intersection_normal_z_resize = intersection_normal_z.unsqueeze_(2)
    intersection_normal = torch.cat((intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 2)
    intersection_normal = intersection_normal / torch.unsqueeze(torch.norm(intersection_normal, p=2, dim=2), 2).repeat(1, 1, 3)

    # Create the point light
    shading = 0
    light_position = camera.repeat(width, height, 1)
    light_norm = torch.unsqueeze(torch.norm(light_position - intersection_pos, p=2, dim=2), 2).repeat(1, 1, 3)
    light_direction_point = (light_position - intersection_pos) / light_norm
    light_direction = camera.repeat(width, height, 1)
    l_dot_n = torch.sum(light_direction * intersection_normal, 2).unsqueeze_(2)
    shading += 2 * torch.max(l_dot_n, Tensor(width, height, 1).fill_(0))[:,:,0] / torch.pow(torch.sum((light_position - intersection_pos) * light_direction_point, dim=2), 2) 

    # Get the final image 
    image = shading * mask  
    image[mask == 0] = 1
    mask = torch.clamp(image * 10000, 0, 1)

    return image, mask

# The energy E captures the difference between a rendered image and
# a desired target image, and the rendered image is a function of the
# SDF values. You could write E(SDF) = ||rendering(SDF)-target_image||^2.
# In addition, there is a second term in the energy as you observed that
# constrains the length of the normal of the SDF to 1. This is a regularization
# term to make sure the output is still a valid SDF.
def loss_fn(output, target, grid, voxel_size, grid_res_x, grid_res_y, grid_res_z, width, height):
    
    image_loss = torch.sum(torch.abs(target - output)) #/ (width * height)

    [grid_normal_x, grid_normal_y, grid_normal_z] = get_grid_normal(grid, voxel_size, grid_res_x, grid_res_y, grid_res_z)
    sdf_loss = torch.sum(torch.abs(torch.pow(grid_normal_x[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                 + torch.pow(grid_normal_y[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2)\
                                 + torch.pow(grid_normal_z[1:grid_res_x-1, 1:grid_res_y-1, 1:grid_res_z-1], 2) - 1)) #/ ((grid_res-1) * (grid_res-1) * (grid_res-1))

    return image_loss, sdf_loss

def sdf_diff(sdf1, sdf2):
    return torch.sum(torch.abs(sdf1 - sdf2)).item()

