import numpy as np
import networkx as nx
import os
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from typing import List, Tuple
import pickle
from tqdm import tqdm
from copy import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from scipy.interpolate import griddata
import torch
from torch_geometric.data import Data
from shapely.geometry import Polygon
import xarray as xr
# from meshkernel import MeshKernel, Mesh2d, GeometryList, OrthogonalizationParameters, ProjectToLandBoundaryOption, MeshRefinementParameters

import sys
sys.path.append('..\\')
from utils.visualization import BasePlotMap

def center_grid_graph(dim1, dim2):
    '''
    Create graph from a rectangular grid of dimensions dim1 x dim2
    Returns networkx graph connecting the grid centers and corresponding 
    node positions
    ------
    dim1: int
        number of grids in the x direction
    dim2: int
        number of grids in the y direction
    '''
    G = nx.grid_2d_graph(dim1, dim2, create_using=nx.DiGraph)
    # for the position, it is assumed that they are located in the centre of each grid
    pos = {i:(x+0.5,y+0.5) for i, (x,y) in enumerate(G.nodes())}
    
    #change keys from (x,y) format to i format
    mapping = dict(zip(G, range(0, G.number_of_nodes())))
    G = nx.relabel_nodes(G, mapping)

    return G, pos

def corners_grid_graph(dim1, dim2):
    '''
    Create graph from a rectangular grid of dimensions dim1 x dim2
    Returns networkx graph connecting the grid corners and corresponding 
    node positions
    ------
    dim1: int
        number of grid points in the x direction
    dim2: int
        number of grid points in the y direction
    '''
    G = nx.grid_2d_graph(dim1, dim2, create_using=nx.DiGraph)
    pos = {i:(x,y) for i, (x,y) in enumerate(G.nodes())}
    
    #change keys from (x,y) format to i format
    mapping = dict(zip(G, range(0, G.number_of_nodes())))
    G = nx.relabel_nodes(G, mapping)

    return G, pos

def get_coords(pos):
    '''
    Returns array of dimensions (n_nodes, 2) containing x and y coordinates of each node
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''
    return np.array([xy for xy in pos.values()])
	
def get_KNN_graph(pos, K=4):
    '''
    Create K-nearest neighbours graph, given the position of the nodes
    '''
    coordinates = get_coords(pos)
    KNN = kneighbors_graph(coordinates, K, mode='connectivity', include_self=False)
    rows, cols = KNN.nonzero()
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.DiGraph()
    G.add_nodes_from(pos)
    G.add_edges_from(edges)

    return G

def get_radius_graph(pos, max_radius=4):
    '''
    Create K-nearest neighbours graph, given the position of the nodes
    '''
    coordinates = get_coords(pos)
    radius_graph = radius_neighbors_graph(coordinates, max_radius, mode='connectivity', include_self=False)
    rows, cols = radius_graph.nonzero()
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.DiGraph()
    G.add_nodes_from(pos)
    G.add_edges_from(edges)

    return G

def get_barycenter(x, y):
    '''Returns barycenter given x and y coordinates'''
    assert x.shape == y.shape, f"Input x and y have incompatible dimensions \n\
                                x: {x.shape}, y: {y.shape}"
    
    if x.ndim == 1:
        length = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
    elif x.ndim == 2:
        length = x.shape[1]
        sum_x = np.sum(x, 1)
        sum_y = np.sum(y, 1)
    else:
        raise ValueError("The dimension of the arrays is wrong")

    return sum_x/length, sum_y/length

def create_mesh_triangle(vertices, segments=None, holes=None, max_area=5, max_smallest_mesh_angle=30):
    '''Creates a mesh using Triangle'''

    if max_smallest_mesh_angle>34:
        raise ValueError("Mesh not computed. Triangle doesn't like hard restrictions.")

    mesh_inputs = {'vertices': vertices}
    
    if segments is not None:
        mesh_inputs['segments'] = segments
    if holes is not None:
        mesh_inputs['holes'] = holes
        
    mesh = tr.triangulate(mesh_inputs, f'cq{max_smallest_mesh_angle}pena{max_area}iD')

    return mesh

def create_mesh(polygon_file='random_polygon.pol', max_refinement_iterations=3, orthogonalize_mesh=False):
    '''Creates a mesh using meshkernel'''
    # mesh = create_mesh_triangle(vertices, segments=None, holes=None, max_area=max_area, max_smallest_mesh_angle=30)
    # mesh2d = Mesh2d(node_x=np.array(mesh['vertices'][:,0], dtype=np.float64),
    #             node_y=np.array(mesh['vertices'][:,1], dtype=np.float64),
    #             edge_nodes=mesh['edges'].ravel())
    # mk = MeshKernel()
    # mk.mesh2d_set(mesh2d)

    with open(polygon_file) as file:
        xy = np.array([[value for value in line.strip().split(",")] for line in file.readlines()[2:]], dtype=np.double)
        # num_points_pol = xy.shape[0]
        # xy = equidistant_perimiter(xy)
    
    # if num_points_pol < 9:
    #     max_refinement_iterations = 5
    # elif num_points_pol < 11:
    #     max_refinement_iterations = 4
    # else:        
    #     max_refinement_iterations = 3

    land_boundary_x = np.array(list(xy[:,0]), dtype=np.double)
    land_boundary_y = np.array(list(xy[:,1]), dtype=np.double)
    polygon = land_boundary = GeometryList(land_boundary_x, land_boundary_y)

    mk = MeshKernel()
    mk.mesh2d_make_mesh_from_polygon(polygon)
    refinement_parameters = MeshRefinementParameters(refine_intersected=True, min_edge_size=0, 
                                                     max_refinement_iterations=max_refinement_iterations, 
                                                     smoothing_iterations=5)
    mk.mesh2d_refine_based_on_polygon(polygon, refinement_parameters)

    if orthogonalize_mesh:
        mk.mesh2d_compute_orthogonalization(ProjectToLandBoundaryOption(0), OrthogonalizationParameters(
            outer_iterations=25, boundary_iterations=25, inner_iterations=25, 
            orthogonalization_to_smoothing_factor=0.95),
            polygon, land_boundary)

    mk.mesh2d_delete_small_flow_edges_and_small_triangles(
        small_flow_edges_length_threshold=0.1, min_fractional_area_triangles=2)

    output_mesh2d = mk.mesh2d_get()

    output_mesh2d.mesh_nodes = np.stack((output_mesh2d.node_x, output_mesh2d.node_y), -1)
    output_mesh2d.dual_nodes = np.stack((output_mesh2d.face_x, output_mesh2d.face_y), -1)
    # output_mesh2d.face_nodes = get_face_nodes_mesh(output_mesh2d)

    return output_mesh2d

def get_face_nodes_mesh(mesh):
    num_faces = mesh.face_x.shape[0]
    max_nodes_per_face = mesh.nodes_per_face.max()
    face_nodes = np.zeros((num_faces, max_nodes_per_face)) * np.nan
    node_position = 0

    for i, num_nodes in enumerate(mesh.nodes_per_face):
        face_nodes[i,:num_nodes] = mesh.face_nodes[node_position : (node_position + num_nodes)]
        node_position += num_nodes

    return face_nodes

def save_mesh(mesh, mesh_file, grid_size):
    '''Saves mesh as NETCDF file using ugrid'''
    # from ugrid import UGrid

    # with UGrid(mesh_file, "w+") as ug:
    #     mesh.node_x = mesh.node_x*grid_size
    #     mesh.node_y = mesh.node_y*grid_size

    #     # 1. Convert a meshkernel mesh2d to an ugrid mesh2d
    #     mesh2d_ugrid = ug.from_meshkernel_mesh2d_to_ugrid_mesh2d(mesh2d=mesh, name="Mesh2d", is_spherical=False)

    #     # 2. Define a new mesh2d
    #     topology_id = ug.mesh2d_define(mesh2d_ugrid)

    #     # 3. Put a new mesh2d
    #     ug.mesh2d_put(topology_id, mesh2d_ugrid)

    #     # 4. Add crs to file
    #     attribute_dict = {
    #         "name": "Unknown projected",
    #         "epsg": np.array([0], dtype=int),
    #         "grid_mapping_name": "Unknown projected",
    #         "longitude_of_prime_meridian": np.array([0.0], dtype=float),
    #         "semi_major_axis": np.array([6378137.0], dtype=float),
    #         "semi_minor_axis": np.array([6356752.314245], dtype=float),
    #         "inverse_flattening": np.array([6356752.314245], dtype=float),
    #         "EPSG_code": "EPSG:0",
    #         "value": "value is equal to EPSG code"}
    #     ug.variable_int_with_attributes_define("projected_coordinate_system", attribute_dict)

    from netCDF4 import Dataset

    if os.path.exists(mesh_file): os.remove(mesh_file)

    with Dataset(mesh_file, mode='w', format='NETCDF4') as ncfile:
        ncfile.createDimension('nNetNode', mesh.node_x.shape[0])
        ncfile.createDimension('nNetLink', mesh.edge_nodes.shape[0]//2)
        ncfile.createDimension('nNetLinkPts', 2)

        ncfile.createVariable('projected_coordinate_system', 'int32', ())
        NetNode_x = ncfile.createVariable('NetNode_x', 'f8', ('nNetNode',))
        NetNode_y = ncfile.createVariable('NetNode_y', 'f8', ('nNetNode',))
        NetNode_z = ncfile.createVariable('NetNode_z', 'f8', ('nNetNode',))
        NetLink = ncfile.createVariable('NetLink', 'int32', ('nNetLink', 'nNetLinkPts'))
        NetLinkType = ncfile.createVariable('NetLinkType', 'int32', ('nNetLink'))

        NetNode_x.units = "m"
        NetNode_x.long_name = "x-coordinate"
        NetNode_y.units = "m"
        NetNode_y.long_name = "y-coordinate"
        NetNode_x.coordinates = "NetNode_x NetNode_y"
        NetNode_y.coordinates = "NetNode_x NetNode_y"
        NetNode_z.coordinates = "NetNode_x NetNode_y"

        NetNode_x[:] = mesh.node_x*grid_size
        NetNode_y[:] = mesh.node_y*grid_size
        NetNode_z[:] = np.zeros_like(mesh.node_y)
        NetLink[:] = mesh.edge_nodes.reshape(-1,2)+1
        NetLinkType[:] = np.ones(mesh.edge_nodes.shape[0]//2, dtype=np.int32)-1

def get_polygon_area(x, y):
    '''Apply shoelace algorithm to evaluate area defined by sequence of points (x,y)'''
    assert x.shape == y.shape, f"Input x and y have incompatible dimensions \n\
                                x: {x.shape}, y: {y.shape}"
    if x.ndim == 1:
        area = 0.5*np.abs(np.dot(x,np.roll(y,1,axis=-1))
            -np.dot(y,np.roll(x,1,axis=-1)))
    elif x.ndim == 2:
        area = 0.5*np.abs(np.multiply(x,np.roll(y,1,axis=-1)).sum(1)
            -np.multiply(y,np.roll(x,1,axis=-1)).sum(1))
    else:
        raise ValueError(f"Input x and y have incorrect dimension ({x.shape})")
    return area

class Mesh(object):
    def __init__(self):
        '''Mixed-elements mesh base object'''
        self.added_ghost_cells = False
        self.node_x = np.array([])
        self.face_x = np.array([])
        self.edge_index = np.array([])

    def _import_from_netcdf(self, nc_file):
        nc_dataset = xr.open_dataset(nc_file)
        self.node_x = nc_dataset['Mesh2d_node_x'].data
        self.node_y = nc_dataset['Mesh2d_node_y'].data

        self.face_x = nc_dataset['Mesh2d_face_x'].data
        self.face_y = nc_dataset['Mesh2d_face_y'].data

        self.edge_index = nc_dataset['Mesh2d_edge_nodes'].data.T - 1
        self.edge_type = nc_dataset['Mesh2d_edge_type'].data # 1:normal edges, 2:BC_edge, 3:other BC_edges
        nc_dataset['Mesh2d_edge_faces'].data[nc_dataset['Mesh2d_edge_faces'].to_masked_array().mask] = 0
        self.dual_edge_index = nc_dataset['Mesh2d_edge_faces'].data.T.astype(int) - 1

        self.face_nodes = nc_dataset['Mesh2d_face_nodes'] - 1
         # mixed mesh
        if isinstance(self.face_nodes.to_masked_array().mask, np.ndarray):
            self.nodes_per_face = (~self.face_nodes.to_masked_array().mask).sum(1).astype(int)
            self.face_nodes = self.face_nodes.data[~self.face_nodes.to_masked_array().mask].astype(int)
        # triangular or quadrilateral mesh
        else:
            self.nodes_per_face = np.ones_like(self.face_nodes).sum(1).data.astype(int)
            self.face_nodes = self.face_nodes.reshape(-1).data.astype(int)
            
        # self.edge_index_BC = self.edge_index[:,self.edge_type == 2].T
        # self.edge_BC = np.where((self.edge_index_BC == self.edge_index.T).sum(1) == 2)[0]

        face_bnd_mask = self.dual_edge_index[0,:] == -1
        self.face_BC = self.dual_edge_index[1,face_bnd_mask]

        extra_face_bnd_mask = self.dual_edge_index[1,:] == -1
        self.extra_face_BC = self.dual_edge_index[0,extra_face_bnd_mask]

        total_face_bnd_mask = extra_face_bnd_mask | face_bnd_mask
        self.dual_edge_index = self.dual_edge_index[:,~total_face_bnd_mask]

    def _get_derived_attributes(self):
        # Nodes
        self.node_xy = np.stack((self.node_x, self.node_y),-1)
        
        # Edges        
        self.edge_relative_distance = self.node_xy[self.edge_index[1,:]] - self.node_xy[self.edge_index[0,:]]
        self.edge_length = np.linalg.norm(self.edge_relative_distance, axis=1)

        self.edge_outward_normal = self.edge_relative_distance/self.edge_length[:,None]
        self.edge_outward_normal[:,1] = -self.edge_outward_normal[:,1]

        # Faces
        self.face_xy = np.stack((self.face_x, self.face_y),-1)

        node_position = 0
        face_areas = []
        for num_nodes in self.nodes_per_face:
            face_node = self.face_nodes[node_position : (node_position + num_nodes)]
            face_nodes_x = self.node_x[face_node]
            face_nodes_y = self.node_y[face_node]
            node_position += num_nodes
            face_area = get_polygon_area(face_nodes_x, face_nodes_y)
            face_areas.append(face_area)
        self.face_area = np.array(face_areas)

    def _import_DEM(self, DEM_file):
        DEM = np.loadtxt(DEM_file)
        ####################### Normalization
        # Calculate the minimum and maximum elevation values
        min_elevation = np.min(DEM[:, 2])
        max_elevation = np.max(DEM[:, 2])
    
        # Normalize the elevation values
        normalized_elevation = (DEM[:, 2] - min_elevation) / (max_elevation - min_elevation)
    
        # Combine the normalized elevation values with x and y coordinates
        normalized_DEM = np.column_stack((DEM[:, 0], DEM[:, 1], normalized_elevation))
        ############################

        #If you want to normalize DEM use this line:
        self.DEM = interpolate_variable(self.face_xy, normalized_DEM[:, :2], normalized_DEM[:, 2], method='nearest')
        #If you want the normal DEM use this:
        #self.DEM = interpolate_variable(self.face_xy, DEM[:,:2], DEM[:,2], method='nearest')

    def __repr__(self) -> str:
        return 'Mesh object with {} nodes, {} edges, and {} faces'.format(
            self.node_x.shape[0], self.edge_index.shape[1], self.face_x.shape[0])

def get_corners(pos):
    '''
    Returns the coordinates of the corners of a grid
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''    
    BL = min(pos.values()) #bottom-left
    TR = max(pos.values()) #top-right
    BR = (BL[0], TR[1]) #bottom-right
    TL = (TR[0], BL[1]) #top-left
    
    return BL, TR, BR, TL

def get_contour(pos):
    '''
    Returns a dictionary with the contours of a grid
    ------
    pos: dict
        keys: (x,y) index of every node
        values: spatial x and y positions of each node
    '''
    BL, TR, BR, TL = get_corners(pos)
    
    x_pos = np.arange(BL[0], TR[0]+1)
    y_pos = np.arange(BL[1], TR[1]+1)
    
    bottom = [(x, BL[1]) for x in x_pos]
    left = [(BL[0], y) for y in y_pos]
    right = [(TR[0], y) for y in y_pos]
    top = [(x, TR[1]) for x in x_pos]
    
    contour = {}

    for point in (bottom + left + right + top):
        key = list(pos.keys())[list(pos.values()).index(point)]
        contour[point] = pos[key]
    
    return contour

def reorder_dict(dictt):
    '''
    Change the key of a dictionary and sorts it by values order
    '''
    new_dict = {}
    
    #sort to exclude double values and order it
    dictt = dict(sorted(dictt.items()))

    #change keys from (x,y) format to i format
    for i, key in enumerate(dictt.keys()):
        new_dict[i] = dictt[key]
        
    return new_dict

def interpolate_variable(mesh_nodes, grid_nodes, variable, method='nearest'):
    '''
    Interpolate variable at specific mesh_nodes contained in grid_nodes
    ------
    *_nodes: np.array
        x and y positions of each node
    variable: np.array
        value of a variable for each point in the domain
    method: str
        choose from 'nearest', 'linear', 'cubic' (see scipy.interpolate.griddata documentation)
    '''
    if isinstance(grid_nodes, dict):
        grid_nodes = get_coords(grid_nodes)

    interpolated_variable = griddata(grid_nodes, variable, mesh_nodes, method=method)
    
    mask = np.isnan(interpolated_variable)
    interpolated_variable[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), interpolated_variable[~mask])

    return interpolated_variable

def interpolate_temporal_variable(mesh_nodes, grid_nodes, temporal_variable, method='nearest'):
    '''Interpolate temporal variable at specific mesh_nodes contained in grid_nodes'''
    total_time = temporal_variable.shape[0]

    interpolated_time_variable = np.array([interpolate_variable(mesh_nodes, grid_nodes, temporal_variable[time_step], method=method) for time_step in range(total_time)])
    
    return interpolated_time_variable

def graph_to_mesh_interpolation_error(original, interpolated, pos, graph, mesh, method='linear'):
    '''Returns difference between original grid and interpolated one and two plots to visualize it:
    plot 1) Spatial distribution of the errors
    plot 2) Histogram distribution of the errors
        If the distribution is not close to a normal, the sampling is biased
    '''
    fig, axs = plt.subplots(1,2, figsize=(18,6))

    grid_nodes = get_coords(pos)
    mesh_nodes = mesh.mesh2d_nodes
    back_to_grid = griddata(mesh_nodes, interpolated, grid_nodes, method=method)

	# Spatial distribution
    diff = (original - back_to_grid).numpy()
    max_diff = np.abs(diff).max()
    Plot = BasePlotMap(diff, pos=pos, graph=graph,
            difference_plot=True, vmin=-max_diff, vmax=max_diff)
    Plot.plot_map(axs[0])
    axs[0].set_title('Errors spatial distribution')
    
	#Histograms
    axs[1].set_title('Errors distribution')
    axs[1].hist(diff[diff!=0], bins=25);
    
    return None

def add_BC_edge_index(edge_index, node_BC, undirected_BC=False):
    """
    Adds ghost nodes to existing graph in correspondance of boundary condition (BC) nodes
    node_BC WILL BE MODIFIED TO THE ACTUAL GHOST NODE DURING THIS OPERATION
    if undirected_BC, the information flow can go also to ghost nodes
    node_BC: torch.Tensor
        contains a list of boundary nodes (nodes with boundary conditions)
    """
    num_nodes = edge_index.max() + 1
    edge_index_BC = []
    node_BC_copy = node_BC.clone()

    for i, node in enumerate(node_BC_copy):
        edge_index_BC.append([num_nodes+i, node])
        node_BC[i] = num_nodes+i
        if undirected_BC:
            edge_index_BC.append([node, num_nodes+i])

    new_edge_index = np.concatenate((edge_index, np.array(edge_index_BC).T), 1)

    return new_edge_index

def get_BC_edge_mesh(mesh):
    BC_edge_index = []
    the_other_node = [] # the node which is not in the BC edge

    for face in mesh.face_BC:
        face_nodes = get_face_nodes_mesh(mesh)[face,:mesh.nodes_per_face[face]].astype(int)

        # Add first element to simplify circular iterations
        face_nodes = np.concatenate((face_nodes, face_nodes[:1]))

        face_BC_dual_edge_index = mesh.dual_edge_index[:,np.where(mesh.dual_edge_index == face)[1]] 
        face_BC_neighbours = face_BC_dual_edge_index[face_BC_dual_edge_index != face]
        face_nodes_BC_neighbours = get_face_nodes_mesh(mesh)[face_BC_neighbours,:mesh.nodes_per_face[face]]

        for i in range(mesh.nodes_per_face[face]):
            is_BC_edge = not any([face_nodes[i] in neighbour and face_nodes[i+1] in neighbour \
                            for neighbour in face_nodes_BC_neighbours])
            if is_BC_edge:
                BC_edge_index.append(face_nodes[i:i+2])
                the_other_node.append(face_nodes[(i-1)%mesh.nodes_per_face[face]]) # VALID ONLY FOR TRIANGLES

    return np.array(BC_edge_index), np.array(the_other_node)

def add_BC_coords(coords, node_BC, number_grids, type_grid='grid'):
    '''Updates the coordinate tensor by addding a ghost cell in correspondence of the boundary node'''

    if type_grid == 'grid':
        if coords[node_BC][0,0] == 0.5:
            coords = torch.cat((coords, torch.stack((torch.tensor(-0.5), coords[node_BC][0,1])).unsqueeze(0)))
        elif coords[node_BC][0,0] == number_grids - 0.5:
            coords = torch.cat((coords, torch.stack((torch.tensor(number_grids + 0.5), coords[node_BC][0,1])).unsqueeze(0)))
        elif coords[node_BC][0,1] == 0.5:
            coords = torch.cat((coords, torch.stack((coords[node_BC][0,0], torch.tensor(-0.5))).unsqueeze(0)))
        elif coords[node_BC][0,1] == number_grids - 0.5:
            coords = torch.cat((coords, torch.stack((coords[node_BC][0,0], torch.tensor(number_grids + 0.5))).unsqueeze(0)))

    elif type_grid == 'mesh':
        raise NotImplementedError("")

    else:
        raise ValueError("'type_grid' must be either 'grid' or 'mesh'")

    return coords

def add_ghost_cells_mesh(mesh):
    _, the_other_node = get_BC_edge_mesh(mesh)

    face_BC_xy = mesh.face_xy[mesh.face_BC]
    node_BC_xy = mesh.node_xy[the_other_node]
    edge_BC_mid = mesh.node_xy[mesh.edge_index_BC].mean(1)

    distance_face_edge_BC = np.linalg.norm(face_BC_xy - edge_BC_mid)
    distance_node_edge_BC = np.linalg.norm(node_BC_xy - edge_BC_mid)

    normal_adapter = np.int32([1, 0])
    ghost_face_BC_xy = edge_BC_mid + mesh.edge_outward_normal[mesh.edge_BC][:,normal_adapter]*distance_face_edge_BC
    ghost_node_BC_xy = edge_BC_mid + mesh.edge_outward_normal[mesh.edge_BC][:,normal_adapter]*distance_node_edge_BC

    mesh.node_x = np.concatenate((mesh.node_x, ghost_node_BC_xy[:,0]))
    mesh.node_y = np.concatenate((mesh.node_y, ghost_node_BC_xy[:,1]))

    mesh.face_x = np.concatenate((mesh.face_x, ghost_face_BC_xy[:,0]))
    mesh.face_y = np.concatenate((mesh.face_y, ghost_face_BC_xy[:,1]))

    mesh.nodes_per_face = np.concatenate((mesh.nodes_per_face, mesh.nodes_per_face[mesh.face_BC]))
    for edge in mesh.edge_index_BC:
        mesh.face_nodes = np.concatenate((mesh.face_nodes, np.array(mesh.face_nodes.max()+1).reshape(-1), edge))

    mesh.added_ghost_cells = True

    return mesh

def add_BC_nodes(edge_index, coords, node_BC, DEM, WD, VX, VY,  
                 undirected_BC=False, type_grid='grid'):
    node_BC_update = torch.tensor(node_BC).clone()

    edge_index = torch.LongTensor(add_BC_edge_index(edge_index, node_BC_update, undirected_BC))
    
    # This part here is terrible but it will get better with meshes
    number_grids = int(DEM.shape[0]**0.5)
    coords = add_BC_coords(coords, node_BC, number_grids, type_grid)

    num_nodes = WD.shape[0]
    num_time_steps = WD.shape[1]
    num_BC_nodes = len(node_BC)
    num_total_nodes = num_nodes + num_BC_nodes
    
    DEM_BC = torch.zeros((num_total_nodes), dtype=torch.float32)
    WD_BC = torch.zeros((num_total_nodes, num_time_steps), dtype=torch.float32)
    VX_BC = torch.zeros((num_total_nodes, num_time_steps), dtype=torch.float32)
    VY_BC = torch.zeros((num_total_nodes, num_time_steps), dtype=torch.float32)

    DEM_BC[:num_nodes] = torch.FloatTensor(DEM)
    WD_BC[:num_nodes,:] = torch.FloatTensor(WD)
    VX_BC[:num_nodes,:] = torch.FloatTensor(VX)
    VY_BC[:num_nodes,:] = torch.FloatTensor(VY)
    
    DEM_BC[-num_BC_nodes:] = torch.FloatTensor(DEM[node_BC])
    WD_BC[-num_BC_nodes:,:] = torch.FloatTensor(WD[node_BC])
    VX_BC[-num_BC_nodes:,:] = torch.FloatTensor(VX[node_BC])
    VY_BC[-num_BC_nodes:,:] = torch.FloatTensor(VY[node_BC])

    return DEM_BC, WD_BC, VX_BC, VY_BC, edge_index, coords, node_BC_update

def add_ghost_cells_attributes(mesh, DEM, WD, VX, VY):
    '''Corrects attribute value at ghost cells'''
    assert mesh.added_ghost_cells, "This function must be executed after add_ghost_cells_mesh"

    num_nodes = WD.shape[0]
    num_time_steps = WD.shape[1]

    num_BC_nodes = mesh.face_BC.shape[0]
    num_total_nodes = mesh.face_x.shape[0]
    
    DEM_BC = torch.zeros((num_total_nodes), dtype=torch.float32)
    WD_BC = torch.zeros((num_total_nodes, num_time_steps), dtype=torch.float32)
    VX_BC = torch.zeros((num_total_nodes, num_time_steps), dtype=torch.float32)
    VY_BC = torch.zeros((num_total_nodes, num_time_steps), dtype=torch.float32)

    DEM_BC[:num_nodes] = torch.FloatTensor(DEM)
    WD_BC[:num_nodes,:] = torch.FloatTensor(WD)
    VX_BC[:num_nodes,:] = torch.FloatTensor(VX)
    VY_BC[:num_nodes,:] = torch.FloatTensor(VY)
    
    DEM_BC[-num_BC_nodes:] = torch.FloatTensor(DEM[mesh.face_BC])
    WD_BC[-num_BC_nodes:,:] = torch.FloatTensor(WD[mesh.face_BC])
    VX_BC[-num_BC_nodes:,:] = torch.FloatTensor(VX[mesh.face_BC])
    VY_BC[-num_BC_nodes:,:] = torch.FloatTensor(VY[mesh.face_BC])

    return DEM_BC, WD_BC, VX_BC, VY_BC

def convert_grid_to_pyg(graph, pos, DEM, WD, VX, VY, hydrograph, 
                        undirected_BC=False, grid_size=100):
    '''Converts a graph or mesh into a PyTorch Geometric Data type 
    Then, add position, DEM, and water variables to data object'''
    DEM = DEM.reshape(-1)

    data = Data()

    edge_index = torch.LongTensor(list(graph.edges)).t().contiguous()
    coords = torch.FloatTensor(get_coords(pos))
    node_BC = np.where(WD[0,:] != 0)[0]
    
    data.DEM, data.WD, data.VX, data.VY, \
    data.edge_index, coords, data.node_BC = add_BC_nodes(
            edge_index, coords, node_BC, DEM, WD.T, VX.T, VY.T, 
            undirected_BC, type_grid='grid')
    
    row, col = data.edge_index

    data.edge_relative_distance = coords[col] - coords[row]
    data.edge_distance = torch.norm(data.edge_relative_distance, dim=1)
    data.normal = data.edge_relative_distance/data.edge_distance[:,None]

    data.pos = coords
    data.num_nodes = len(data.DEM)
    data.area = torch.ones_like(data.DEM)*grid_size*grid_size

    data.BC = torch.FloatTensor(hydrograph).unsqueeze(0).repeat(len(node_BC), 1, 1) # This repeats the same BC
    data.type_BC = torch.tensor(2, dtype=torch.int) # FOR NOW ONLY DISCHARGE BC

    return data

def create_grid_dataset(dataset_folder, n_sim, start_sim=1, 
                        number_grids=64, with_hydrograph=False,
                        KNN=0, max_radius=0):
    '''
    Creates a pytorch geometric dataset with n_sim simulations
    returns a regular grid graph dataset
    ------
    dataset_folder: str, path-like
        path to raw dataset location
    n_sim: int
        number of simulations used in the dataset creation
    with_hydrograph: bool
        if True, reads hydrograph file
    KNN: int
        if != 0, cretes K-nearest-neighbours graph
    max_radius: float
        if != 0, cretes max-radius graph
    '''
    grid_dataset = []

    graph, pos = center_grid_graph(number_grids,number_grids)
    
    if KNN != 0:
        graph = get_KNN_graph(pos, KNN)
    
    if max_radius != 0:
        graph = get_radius_graph(pos, max_radius)

    for i in tqdm(range(start_sim,start_sim+n_sim)):

        DEM = np.loadtxt(f"{dataset_folder}\\DEM\\DEM_{i}.txt")[:,2]
        WD = np.loadtxt(f"{dataset_folder}\\WD\\WD_{i}.txt")
        VX = np.loadtxt(f"{dataset_folder}\\VX\\VX_{i}.txt")
        VY = np.loadtxt(f"{dataset_folder}\\VY\\VY_{i}.txt")
        if with_hydrograph:
            hydrograph = np.loadtxt(f"{dataset_folder}\\Hydrograph\\Hydrograph_{i}.txt")
            hydrograph[:,0] /= 60 # convert to minutes
        else:
            hydrograph = torch.FloatTensor([200])
        
        grid_i = convert_grid_to_pyg(graph, pos, DEM, WD, VX, VY, hydrograph)
        grid_dataset.append(grid_i)
    
    return grid_dataset

def create_mesh_dataset(dataset_folder, n_sim, start_sim=1):
    '''
    Creates a list of pytorch geometric Data objects with n_sim simulations
    returns a mesh dataset
    ------
    dataset_folder: str, path-like
        path to raw dataset location
    n_sim: int
        number of simulations used in the dataset creation
    '''
    mesh_dataset = []

    for i in tqdm(range(start_sim,start_sim+n_sim)):
        
        data = Data()

        output_map = f'{dataset_folder}/Results/dr49_{i}_map.nc'
        nc_dataset = xr.open_dataset(output_map)

        WD = nc_dataset['Mesh2d_waterdepth'].data.T
        VX = nc_dataset['Mesh2d_ucx'].data.T
        VY = nc_dataset['Mesh2d_ucy'].data.T
        
        mesh = Mesh()
        mesh._import_from_netcdf(output_map)
        mesh._get_derived_attributes()
        mesh._import_DEM(f"{dataset_folder}\\DEM\\DEM.txt")
        # mesh = add_ghost_cells_mesh(mesh)

        # data.DEM, data.WD, data.VX, data.VY = add_ghost_cells_attributes(mesh, mesh.DEM, WD, VX, VY)
        # mesh.DEM = data.DEM

        # data.node_BC = torch.tensor(mesh.face_BC)
        # mesh.dual_edge_index = add_BC_edge_index(mesh.dual_edge_index, data.node_BC, undirected_BC=False)
        # mesh._get_derived_attributes()
            
        data.DEM = torch.FloatTensor(mesh.DEM)
        data.WD = torch.FloatTensor(WD)
        data.VX = torch.FloatTensor(VX)
        data.VY = torch.FloatTensor(VY)
        
        data.pos = torch.FloatTensor(mesh.face_xy)
        data.edge_index = torch.LongTensor(mesh.dual_edge_index)
        data.edge_distance = torch.FloatTensor(mesh.edge_length[mesh.edge_type < 3])
        data.edge_relative_distance = torch.FloatTensor(mesh.edge_relative_distance[mesh.edge_type < 3])
        data.normal = torch.FloatTensor(mesh.edge_outward_normal[mesh.edge_type < 3])
        data.num_nodes = mesh.face_x.shape[0]
        data.area = torch.FloatTensor(mesh.face_area)

        # data.mesh = mesh

        # hydrograph = np.loadtxt(f"{dataset_folder}\\Hydrograph\\Hydrograph_{i}.txt")
        # hydrograph = torch.FloatTensor([50])
        # # hydrograph[:,0] /= 60 # convert to minutes
        # data.BC = torch.FloatTensor(hydrograph).unsqueeze(0).repeat(len(data.node_BC), 1, 1) # This repeats the same BC
        # data.type_BC = torch.tensor(2, dtype=torch.int) # FOR NOW ONLY DISCHARGE BC

        mesh_dataset.append(data)
    
    return mesh_dataset

def create_dataset_folders(dataset_folder='datasets'):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    train_folder = os.path.join(dataset_folder, 'train')
    test_folder = os.path.join(dataset_folder, 'test')

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)


def save_database(dataset, name, out_path='datasets'):
    '''
    This function saves the geometric database into a pickle file
    The name of the file is given by the type of graph and number of simulations
    ------
    dataset: list
        list of geometric datasets for grid and mesh
    names: str
        name of saved dataset
    out_path: str, path-like
        output file location
    '''
    n_sim = len(dataset)
    path = f"{out_path}/{name}.pkl"
    
    if os.path.exists(path):
        os.remove(path)
    elif not os.path.exists(out_path):
        os.mkdir(out_path)
    
    pickle.dump(dataset, open(path, "wb" ))
        
    return None

def generate_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int, seed: float) -> List[Tuple[float, float]]:
    """
    TAKEN FROM: https://stackoverflow.com/questions/8997099/algorithm-to-generate-random-2d-polygon
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    np.random.seed(seed)

    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * np.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity, seed)

    # now generate the points
    points = []
    angle = np.random.uniform(0, 2 * np.pi)
    for i in range(num_vertices):
        radius = clip(np.random.normal(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * np.cos(angle),
                 center[1] + radius * np.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    polygon = Polygon(points)

    return polygon

def random_angle_steps(steps: int, irregularity: float, seed: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    np.random.seed(seed)

    # generate n angle steps
    angles = []
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = np.random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * np.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def clip(value, lower, upper):
    """
    Given an interval, values outside the interval are clipped to the interval
    edges.
    """
    return min(upper, max(value, lower))

# def generate_random_polygon(max_extent, num_points=50, seed=42):
#     """Generates a polygon as a convex hull of a set of random points"""
#     np.random.seed(seed)
#     from scipy.spatial import ConvexHull

#     points = np.random.random((num_points, 2))*max_extent   # random points in 2-D
#     hull = ConvexHull(points)

#     polygon = Polygon(points[hull.vertices])

#     return polygon

def equidistant_perimiter(vertices):
    segments_lengths = np.array([np.linalg.norm(vertices[i+1,:] - vertices[i,:]) for i in range(len(vertices)-1)])
    min_length = segments_lengths.min()

    new_vertices = copy(vertices)
    for i in range(len(vertices)-1):
        segment_ratio = segments_lengths[i] / min_length
        if segment_ratio > 2:
            more_segments = np.linspace(vertices[i,:], vertices[i+1,:], int(np.ceil(segment_ratio/2)))
            index = i + len(new_vertices) - len(vertices)
            new_vertices = np.concatenate((new_vertices[:index+1,:], more_segments[1:-1], new_vertices[index+1:,:]))

    return new_vertices

def save_polygon_to_file(polygon, filename):
    """Save the generated polygon to a .pol file."""
    with open(filename, 'w') as f:
        f.write(f'# Extent: {polygon.bounds}\n')
        f.write('# Coordinates (x, y):\n')
        for coord in polygon.exterior.coords:
            f.write(f'{coord[0]}, {coord[1]}\n')

def create_dike(dike_corners:list, dike_points_frequency:int=1, dike_width:float=1):
    """Returns an array of points that discretize a linear element defined by its corners"""
    x0, y0 = dike_corners[0]
    x1, y1 = dike_corners[1]

    if y1-y0 > x1-x0:
        if y0 > y1:
            dike_points_frequency *= -1
        y = np.arange(y0, y1, dike_points_frequency)
        x = np.linspace(x0, x1, y.shape[0])

        diagonal1 = np.stack((x, y), 1)
        diagonal2 = np.stack((x+dike_width, y), 1)

    else:
        if x0 > x1:
            dike_points_frequency *= -1
        x = np.arange(x0, x1, dike_points_frequency)
        y = np.linspace(y0, y1, x.shape[0])

        diagonal1 = np.stack((x, y), 1)
        diagonal2 = np.stack((x, y+dike_width), 1)

    return np.concatenate((diagonal1, diagonal2))

def is_point_inside_polygon(point, polygon):
    '''Determine if a point (x,y) is inside a polygon (list of points)'''
    x, y = point
    n = len(polygon)

    for i in range(n):
        inside = False
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if y1 < y and y2 >= y or y2 < y and y1 >= y:
            if x1 + (y - y1) / (y2 - y1) * (x2 - x1) > x:
                inside = True
                break

    return inside

def generate_random_polygon_with_dike(save_polygon=False, avg_radius=100, irregularity=0.5, 
                                      spikiness=0.2, seed=42, num_vertices=20,
                                      dike_corners=None, min_dike_length=0.5, **dike_options):
    """Generates a polygon with inside it a linear dike element"""
    np.random.seed(seed)

    polygon = generate_polygon(center=(avg_radius, avg_radius), avg_radius=avg_radius, 
                               irregularity=irregularity, spikiness=spikiness,
                               num_vertices=num_vertices, seed=seed)

    if save_polygon:
        save_polygon_to_file(polygon, 'random_polygon.pol')

    vertices = np.array(polygon.exterior.coords)
    vertices = equidistant_perimiter(vertices)
    
    if dike_corners is None:
        dike_length = 0
        while dike_length < min_dike_length:
            dike_corners = np.random.random((2,2))
            dike_length = np.linalg.norm((dike_corners[1]-dike_corners[0]))
        dike_corners *= avg_radius*1.5

    dike = create_dike(dike_corners, **dike_options)

    outside_points = np.array([is_point_inside_polygon(point, vertices) for point in dike])

    return vertices, dike[outside_points]

def plot_faces(mesh, ax=None, face_value=None, **kwargs):
    """Plots the mesh with face values if specified"""
    ax = ax or plt.gca()

    node_position = 0
    patches = []
    for num_nodes in mesh.nodes_per_face:
        face_node = mesh.face_nodes[node_position : (node_position + num_nodes)]
        face_nodes_x = mesh.node_x[face_node]
        face_nodes_y = mesh.node_y[face_node]
        face = np.stack((face_nodes_x, face_nodes_y)).T
        node_position += num_nodes
        patches.append(mpl.patches.Polygon(face, closed=True))
        
    collection = PatchCollection(patches, **kwargs)
    collection.set_array(face_value)
    ax.add_collection(collection)
    ax.set_xlim(mesh.node_x.min(), mesh.node_x.max())
    ax.set_ylim(mesh.node_y.min(), mesh.node_y.max())

    return ax