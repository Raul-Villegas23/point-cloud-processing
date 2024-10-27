# Import necessary libraries
import numpy as np
import open3d as o3d
import laspy
import os
import pandas as pd
from scipy.spatial import Delaunay
import trimesh
from pygltflib import GLTF2, BufferFormat

def load_ply_point_cloud(ply_file_path):
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    
    if point_cloud.has_colors():
        print("RGB values found in the .ply file.")
    else:
        print("No RGB values found in the .ply file.")
    return point_cloud

def load_laz_point_cloud(laz_file_path, decimation_factor=10):
    """
    Load a .laz point cloud file into an Open3D point cloud with optional decimation.
    
    Args:
    - laz_file_path: The path to the .laz file.
    - decimation_factor: The factor by which to downsample the point cloud. Default is 1 (no downsampling).
    
    Returns:
    - point_cloud: The loaded Open3D PointCloud object.
    """
    # Open the .laz file using laspy
    with laspy.open(laz_file_path) as las_file:
        las_data = las_file.read()

    # Extract the X, Y, Z coordinates
    xyz = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()

    # Apply decimation by selecting every nth point based on the decimation factor
    if decimation_factor > 1:
        print(f"Original point cloud has {len(xyz)} points.")
        xyz = xyz[::decimation_factor]
        print(f"Decimated point cloud to {len(xyz)} points.")

    # Create the Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)

    # Check for RGB color information and apply decimation if available
    if hasattr(las_data, 'red') and hasattr(las_data, 'green') and hasattr(las_data, 'blue'):
        red = las_data.red[::decimation_factor] / 65535.0
        green = las_data.green[::decimation_factor] / 65535.0
        blue = las_data.blue[::decimation_factor] / 65535.0
        rgb = np.vstack((red, green, blue)).transpose()
        point_cloud.colors = o3d.utility.Vector3dVector(rgb)
    else:
        print("No RGB color information found in .laz file.")

    return point_cloud


def load_point_cloud_from_dataframe(pcd_df):
    """
    Load a point cloud from a Pandas DataFrame and convert it to an Open3D point cloud.
    
    Args:
    - pcd_df (pd.DataFrame): DataFrame with columns 'X', 'Y', 'Z' for coordinates, and optionally 'R', 'G', 'B' for color.
    
    Returns:
    - o3d.geometry.PointCloud: The loaded point cloud in Open3D format.
    """
    # Ensure the data is in the correct format (float)
    points = np.array(pcd_df[['X', 'Y', 'Z']], dtype=np.float64)

    # Create the point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    
    # Check if color information exists in the DataFrame and add it if available
    if all(col in pcd_df.columns for col in ['R', 'G', 'B']):
        colors = np.array(pcd_df[['R', 'G', 'B']], dtype=np.float64) / 255.0  # Normalize colors
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    else:
        print("No RGB color information found in DataFrame.")
    
    return point_cloud

def load_xyz_point_cloud(xyz_file_path):
    """
    Load an .xyz point cloud file into a Pandas DataFrame and convert it to an Open3D point cloud.

    Args:
    - xyz_file_path: The path to the .xyz file

    Returns:
    - o3d.geometry.PointCloud: The loaded point cloud in Open3D format.
    """
    try:
        # Load the file using ';' as the delimiter
        pcd_df = pd.read_csv(xyz_file_path, sep=";", header=0)
        
        # Select only the relevant columns (X, Y, Z, and optionally R, G, B)
        if {'X', 'Y', 'Z'}.issubset(pcd_df.columns):
            if {'R', 'G', 'B'}.issubset(pcd_df.columns):
                # If RGB is available, pass the full XYZRGB DataFrame
                pcd_df = pcd_df[['X', 'Y', 'Z', 'R', 'G', 'B']]
            else:
                # If RGB is not available, use only XYZ columns
                pcd_df = pcd_df[['X', 'Y', 'Z']]
            
            # Convert the DataFrame to a point cloud
            return load_point_cloud_from_dataframe(pcd_df)
        else:
            raise ValueError("The file does not contain the required X, Y, Z columns.")
    
    except Exception as e:
        print(f"Error loading .xyz point cloud: {e}")
        return None

def load_point_cloud(file_path):
    file_extension = os.path.splitext(file_path)[-1].lower()
    if file_extension == ".laz" or file_extension == ".las":
        print("Loading .laz point cloud...")
        return load_laz_point_cloud(file_path, decimation_factor=50)
    elif file_extension == ".xyz":
        print("Loading .xyz point cloud...")
        return load_xyz_point_cloud(file_path)
    elif file_extension == ".ply":
        print("Loading .ply point cloud...")
        return load_ply_point_cloud(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


# New function to perform ball pivoting
def perform_ball_pivoting(point_cloud):
    """
    Estimate normals, calculate radius based on nearest neighbor distances,
    and perform the ball-pivoting algorithm to create a mesh.
    """
    # Step 4: Estimate normals using KD-Tree for efficiency
    if not point_cloud.has_normals():
        print("Estimating normals using KD-Tree...")
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Orient the normals consistently with the tangent plane to avoid issues in meshing
    point_cloud.orient_normals_consistent_tangent_plane(10)
    
    # Step 5: Estimate nearest neighbor distance and calculate radius for ball-pivoting
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3.0 * avg_dist  # Increase radius for faster mesh generation
    print(f"Average neighbor distance = {avg_dist:.6f}")
    
    # Step 6: Create a mesh using the Ball-Pivoting algorithm with the calculated radius
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    double_sided_mesh = create_double_sided_mesh(mesh)
    
    # Step 7: Visualize the mesh with the original point cloud
    print("Mesh generated using ball-pivoting.")
    o3d.visualization.draw_geometries([point_cloud, double_sided_mesh], window_name='Ball-Pivoting Mesh', width=1200, height=800)

    return mesh

# Step 8: Mesh simplification using Quadric Decimation
def quadratic_decimation_simplify_mesh(mesh, target_triangle_count=10000):
    print(f"Original mesh has {len(mesh.triangles)} triangles.")
    simplified_mesh = mesh.simplify_quadric_decimation(target_triangle_count)
    print(f"Simplified mesh to {len(simplified_mesh.triangles)} triangles.")
    return simplified_mesh

def clustering_simplify_with_average(mesh, voxel_size=0.05):
    """
    Simplify a mesh using vertex clustering with average contraction.

    Args:
    - mesh (o3d.geometry.TriangleMesh): The input mesh to be simplified.
    - voxel_size (float): The size of the voxel grid for clustering vertices.

    Returns:
    - simplified_mesh (o3d.geometry.TriangleMesh): The simplified mesh.
    """
    # Apply vertex clustering simplification with average contraction
    simplified_mesh = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    
    return simplified_mesh

def create_double_sided_mesh(mesh):
    """
    Create a double-sided mesh by inverting the faces to create back-facing triangles.
    
    Args:
    - mesh (o3d.geometry.TriangleMesh): The input mesh to be converted to double-sided.
    
    Returns:
    - double_sided_mesh (o3d.geometry.TriangleMesh): The resulting double-sided mesh.
    """
    # Original vertices and faces
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # Invert the faces to create back-facing triangles
    inverted_faces = faces[:, ::-1]

    # Create a new Open3D mesh with the same vertices and inverted faces
    double_sided_mesh = o3d.geometry.TriangleMesh()
    double_sided_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    double_sided_mesh.triangles = o3d.utility.Vector3iVector(inverted_faces)

    # Assign vertex colors if available
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        double_sided_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # Compute vertex normals for visualization
    double_sided_mesh.compute_vertex_normals()
    
    return double_sided_mesh


def convert_open3d_to_trimesh(double_sided_mesh):
    """
    Convert an Open3D mesh to a Trimesh object and set up a double-sided material.

    Args:
    - double_sided_mesh (o3d.geometry.TriangleMesh): The input Open3D mesh to be converted.

    Returns:
    - trimesh_mesh (trimesh.Trimesh): The resulting Trimesh object with double-sided material.
    """
    # Convert Open3D mesh to Trimesh
    vertices = np.asarray(double_sided_mesh.vertices)
    faces = np.asarray(double_sided_mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Set up a double-sided material in Trimesh
    material = trimesh.visual.material.SimpleMaterial()
    material.doubleSided = True
    trimesh_mesh.visual.material = material

    return trimesh_mesh

if __name__ == "__main__":
    # Path to point cloud file (either .laz, .xyz, or .ply)
    input_file_path = "./DATA/"
    file_name = "hoornestr.las"
    point_cloud_file = input_file_path + file_name

    # Load point cloud based on file type
    point_cloud = load_point_cloud(point_cloud_file)
    print(f"Loaded point cloud has {len(point_cloud.points)} points.")
    
    # Voxel downsampling
    voxel_size = 0.05
    point_cloud = point_cloud.voxel_down_sample(voxel_size)
    print(f"Voxel downsampled point cloud has {len(point_cloud.points)} points.")
    

    # Perform ball-pivoting mesh reconstruction
    ball_pivoting_mesh = perform_ball_pivoting(point_cloud)

    # Simplify the mesh using Quadric Decimation
    target_triangle_count = len(ball_pivoting_mesh.triangles) // 5 
    simplified_mesh = quadratic_decimation_simplify_mesh(ball_pivoting_mesh, target_triangle_count)

    # Simplify the mesh using vertex clustering with average contraction
    # simplified_mesh = clustering_simplify_with_average(ball_pivoting_mesh, voxel_size=0.5)


    # Create a double-sided mesh for visualization
    double_sided_mesh = create_double_sided_mesh(simplified_mesh)

    # Visualize the mesh with both front and back faces visible
    o3d.visualization.draw_geometries([double_sided_mesh], window_name='Double-Sided Mesh')

    # Convert Open3D mesh to Trimesh and set up double-sided material
    trimesh_mesh = convert_open3d_to_trimesh(double_sided_mesh)

    # Step 2: Export to glTF using Trimesh (as intermediate format)
    output_glb_path = f"./DATA/RESULTS/{file_name.split('.')[0]}_mesh.glb"
    trimesh_mesh.export(output_glb_path, file_type='glb')