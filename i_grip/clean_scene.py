import numpy as np
import collections
import uuid

from trimesh import caching, convex, grouping, inertia, transformations, units, util
from trimesh import Scene
from trimesh.scene.transforms import SceneGraph
from trimesh.parent import Geometry, Geometry3D
from trimesh.typed import (
    ArrayLike,
    Dict,
    List,
    NDArray,
    Optional,
    Sequence,
    Tuple,
    Union,
    float64,
    int64,
)
from trimesh.scene import cameras, lighting

GeometryInput = Union[
    Geometry, Sequence[Geometry], NDArray[Geometry], Dict[str, Geometry]
]

class CleanSceneGraph(SceneGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('CleanSceneGraph __init__')
        
    def remove_geometries(self, geometries: Union[str, set, Sequence], full_clean =False):
        """
        Remove the reference for specified geometries
        from nodes without deleting the node.

        Parameters
        ------------
        geometries : list or str
          Name of scene.geometry to dereference.
        """
        # make sure we have a set of geometries to remove
        if util.is_string(geometries):
            geometries = [geometries]
        geometries = set(geometries)

        # remove the geometry reference from the node without deleting nodes
        # this lets us keep our cached paths, and will not screw up children
        if full_clean:
            for geom in geometries:
                self.transforms.remove_node(geom)
        else:
            for attrib in self.transforms.node_data.values():
                if "geometry" in attrib and attrib["geometry"] in geometries:
                    attrib.pop("geometry")

        # it would be safer to just run _cache.clear
        # but the only property using the geometry should be
        # nodes_geometry: if this becomes not true change this to clear!
        self._cache.clear()
        print('cache cleared by remove_geometries')
        self.transforms._hash = None
        

class CleanScene(Scene):
    """
    A simple scene graph which can be rendered directly via
    pyglet/openGL or through other endpoints such as a
    raytracer. Meshes are added by name, which can then be
    moved by updating transform in the transform tree.
    """

    def __init__(
        self,
        geometry: Optional[GeometryInput] = None,
        base_frame: str = "world",
        metadata: Optional[Dict] = None,
        graph: Optional[SceneGraph] = None,
        camera: Optional[cameras.Camera] = None,
        lights: Optional[Sequence[lighting.Light]] = None,
        camera_transform: Optional[NDArray] = None,
    ):
        """
        Create a new Scene object.

        Parameters
        -------------
        geometry : Trimesh, Path2D, Path3D PointCloud or list
          Geometry to initially add to the scene
        base_frame
          Name of base frame
        metadata
          Any metadata about the scene
        graph
          A passed transform graph to use
        camera : Camera or None
          A passed camera to use
        lights : [trimesh.scene.lighting.Light] or None
          A passed lights to use
        camera_transform
          Homogeneous (4, 4) camera transform in the base frame
        """
        print('CleanScene __init__')
        # mesh name : Trimesh object
        self.geometry = collections.OrderedDict()

        # create a new graph
        self.graph = CleanSceneGraph(base_frame=base_frame)

        # create our cache
        self._cache = caching.Cache(id_function=self.__hash__)

        if geometry is not None:
            # add passed geometry to scene
            self.add_geometry(geometry)

        # hold metadata about the scene
        self.metadata = {}
        if isinstance(metadata, dict):
            self.metadata.update(metadata)

        if graph is not None:
            # if we've been passed a graph override the default
            self.graph = graph

        if lights is not None:
            self.lights = lights
        if camera is not None:
            self.camera = camera
            if camera_transform is not None:
                self.camera_transform = camera_transform
                
    
    def delete_geometry(self, names: Union[set, str, Sequence], full_clean = True) -> None:
        """
        Delete one more multiple geometries from the scene and also
        remove any node in the transform graph which references it.

        Parameters
        --------------
        name : hashable
          Name that references self.geometry
        """
        print('scene delete_geometry')
        # make sure we have a set we can check
        if util.is_string(names):
            names = [names]
        names = set(names)

        # remove the geometry reference from relevant nodes
        self.graph.remove_geometries(names, full_clean=full_clean)
        # remove the geometries from our geometry store
        [self.geometry.pop(name, None) for name in names]