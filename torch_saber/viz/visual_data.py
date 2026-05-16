import itertools
import os.path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from xml.etree import ElementTree

import mujoco
import numpy as np
import pyvista as pv
from etils import epath


def quat_mul_np(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiplies two quaternions.

    Args:
      u: (4,) quaternion (w,x,y,z)
      v: (4,) quaternion (w,x,y,z)

    Returns:
      A quaternion u * v.
    """
    return np.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ]
    )


def rotate_np(vec: np.ndarray, quat: np.ndarray):
    """Rotates a vector vec by a unit quaternion quat.

    Args:
      vec: (3,) a vector
      quat: (4,) a quaternion

    Returns:
      ndarray(3) containing vec rotated by quat.
    """
    if len(vec.shape) != 1:
        raise ValueError("vec must have no batch dimensions.")
    s, u = quat[0], quat[1:]
    r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
    r = r + 2 * s * np.cross(u, vec)
    return r


def _transform_do(parent_pos: np.ndarray, parent_quat: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = parent_pos + math.rotate_np(pos, parent_quat)
    rot = math.quat_mul_np(parent_quat, quat)
    return pos, rot


def _offset(elem: ElementTree.Element, parent_pos: np.ndarray, parent_quat: np.ndarray):
    """Offsets an element."""
    pos = elem.attrib.get("pos", "0 0 0")
    quat = elem.attrib.get("quat", "1 0 0 0")
    pos = np.fromstring(pos, sep=" ")
    quat = np.fromstring(quat, sep=" ")
    fromto = elem.attrib.get("fromto", None)
    if fromto:
        # fromto attributes are not compatible with pos/quat attributes
        from_pos = np.fromstring(" ".join(fromto.split(" ")[0:3]), sep=" ")
        to_pos = np.fromstring(" ".join(fromto.split(" ")[3:6]), sep=" ")
        from_pos, _ = _transform_do(parent_pos, parent_quat, from_pos, quat)
        to_pos, _ = _transform_do(parent_pos, parent_quat, to_pos, quat)
        fromto = " ".join("%f" % i for i in np.concatenate([from_pos, to_pos]))
        elem.attrib["fromto"] = fromto
        return
    pos, quat = _transform_do(parent_pos, parent_quat, pos, quat)
    pos = " ".join("%f" % i for i in pos)
    quat = " ".join("%f" % i for i in quat)
    elem.attrib["pos"] = pos
    elem.attrib["quat"] = quat


def _fuse_bodies(elem: ElementTree.Element):
    """Fuses together parent child bodies that have no joint."""

    for child in list(elem):  # we will modify elem children, so make a copy
        _fuse_bodies(child)
        # this only applies to bodies with no joints
        if child.tag != "body":
            continue
        if child.find("joint") is not None or child.find("freejoint") is not None:
            continue
        cpos = child.attrib.get("pos", "0 0 0")
        cpos = np.fromstring(cpos, sep=" ")
        cquat = child.attrib.get("quat", "1 0 0 0")
        cquat = np.fromstring(cquat, sep=" ")
        for grandchild in child:
            # TODO: might need to offset more than just these tags
            if grandchild.tag in ("body", "geom", "site", "camera") and (cpos != 0).any():
                _offset(grandchild, cpos, cquat)
            elem.append(grandchild)
        elem.remove(child)


def _get_meshdir(elem: ElementTree.Element) -> Union[str, None]:
    """Gets the mesh directory specified by the mujoco compiler tag."""
    elems = list(elem.iter("compiler"))
    return elems[0].get("meshdir") if elems else None


def _find_assets(
    elem: ElementTree.Element,
    path: epath.Path,
    meshdir: Optional[str],
) -> Dict[str, bytes]:
    """Loads assets from an xml given a base path."""
    assets = {}
    path = path if path.is_dir() else path.parent
    fname = elem.attrib.get("file") or elem.attrib.get("filename")
    if fname and fname.endswith(".xml"):
        # an asset can be another xml!  if so, we must traverse it, too
        asset = (path / fname).read_text()
        asset_xml = ElementTree.fromstring(asset)
        _fuse_bodies(asset_xml)
        asset_meshdir = _get_meshdir(asset_xml)
        assets[fname] = ElementTree.tostring(asset_xml)
        assets.update(_find_assets(asset_xml, path, asset_meshdir))
    elif fname:
        # mesh, png, etc
        path = path / meshdir if meshdir else path
        assets[fname] = (path / fname).read_bytes()

    for child in list(elem):
        assets.update(_find_assets(child, path, meshdir))

    return assets


def _get_name(mj: mujoco.MjModel, i: int) -> str:
    names = mj.names[i:].decode("utf-8")
    return names[: names.find("\x00")]


def _check_custom(mj: mujoco.MjModel, custom: Dict[str, np.ndarray]) -> None:
    """Validates fields in custom."""
    if not (0 <= custom["spring_mass_scale"] <= 1 and 0 <= custom["spring_inertia_scale"] <= 1):
        raise ValueError("Spring inertia and mass scale must be in [0, 1].")
    if "init_qpos" in custom and custom["init_qpos"].shape[0] != mj.nq:
        size = custom["init_qpos"].shape[0]
        raise ValueError(f"init_qpos had length {size} but expected length {mj.nq}.")


def _get_custom(mj: mujoco.MjModel) -> Dict[str, np.ndarray]:
    """Gets custom mjcf parameters for brax, with defaults."""
    default = {
        "ang_damping": (0.0, None),
        "vel_damping": (0.0, None),
        "baumgarte_erp": (0.1, None),
        "spring_mass_scale": (0.0, None),
        "spring_inertia_scale": (0.0, None),
        "joint_scale_pos": (0.5, None),
        "joint_scale_ang": (0.2, None),
        "collide_scale": (1.0, None),
        "matrix_inv_iterations": (10, None),
        "solver_maxls": (20, None),
        "elasticity": (0.0, "geom"),
        "constraint_stiffness": (2000.0, "body"),
        "constraint_limit_stiffness": (1000.0, "body"),
        "constraint_ang_damping": (0.0, "body"),
        "constraint_vel_damping": (0.0, "body"),
    }

    # add user provided overrides to the defaults
    for i, ni in enumerate(mj.name_numericadr):
        nsize = mj.numeric_size[i]
        name = _get_name(mj, ni)
        val = mj.numeric_data[mj.numeric_adr[i] : mj.numeric_adr[i] + nsize]
        typ = default[name][1] if name in default else None
        default[name] = (val, typ)

    # gather custom overrides with correct sizes
    custom = {}
    for name, (val, typ) in default.items():
        val = np.array([val])
        size = {
            "body": mj.nbody - 1,  # ignore the world body
            "geom": mj.ngeom,
        }.get(typ, val.shape[-1])
        if val.shape[-1] != size and val.shape[-1] > 1:
            # the provided shape does not match against our default size
            raise ValueError(f'"{name}" custom arg needed {size} values for the "{typ}" type, ' f"but got {val.shape[-1]} values.")
        elif val.shape[-1] != size and val.shape[-1] == 1:
            val = np.repeat(val, size)
        val = val.squeeze() if not typ else val.reshape(size)
        if typ == "body":
            # pad one value for the world body, which gets dropped at Link creation
            val = np.concatenate([[val[0]], val])
        custom[name] = val

    # get tuple custom overrides
    for i, ni in enumerate(mj.name_tupleadr):
        start, end = mj.tuple_adr[i], mj.tuple_adr[i] + mj.tuple_size[i]
        objtype = mj.tuple_objtype[start:end]
        name = _get_name(mj, ni)
        if not all(objtype[0] == objtype):
            raise NotImplementedError(f'All tuple elements "{name}" should have the same object type.')
        if objtype[0] not in [1, 5]:
            raise NotImplementedError(f'Custom tuple "{name}" with objtype=={objtype[0]} is not supported.')
        typ = {1: "body", 5: "geom"}[objtype[0]]
        if name in default and default[name][1] != typ:
            raise ValueError(f'Custom tuple "{name}" is expected to be associated with' f" the {default[name][1]} objtype.")

        size = {1: mj.nbody, 5: mj.ngeom}[objtype[0]]
        default_val, _ = default.get(name, (0.0, None))
        arr = np.repeat(default_val, size)
        objid = mj.tuple_objid[start:end]
        objprm = mj.tuple_objprm[start:end]
        arr[objid] = objprm
        custom[name] = arr

    _check_custom(mj, custom)
    return custom


def validate_model(mj: mujoco.MjModel) -> None:
    """Checks if a MuJoCo model is compatible with brax physics pipelines."""
    if mj.opt.integrator != 0:
        raise NotImplementedError("Only euler integration is supported.")
    if mj.opt.cone != 0:
        raise NotImplementedError("Only pyramidal cone friction is supported.")
    if (mj.geom_fluid != 0).any():
        raise NotImplementedError("Ellipsoid fluid model not implemented.")
    if mj.opt.wind.any():
        raise NotImplementedError("option.wind is not implemented.")
    if mj.opt.impratio != 1:
        raise NotImplementedError("Only impratio=1 is supported.")

    # actuators
    if any(i not in [0, 1] for i in mj.actuator_biastype):
        raise NotImplementedError("Only actuator_biastype in [0, 1] are supported.")
    if any(i != 0 for i in mj.actuator_gaintype):
        raise NotImplementedError("Only actuator_gaintype in [0] is supported.")
    if not (mj.actuator_trntype == 0).all():
        raise NotImplementedError("Only joint transmission types are supported for actuators.")

    # solver parameters
    if (mj.geom_solmix[0] != mj.geom_solmix).any():
        raise NotImplementedError("geom_solmix parameter not supported.")
    if (mj.geom_priority[0] != mj.geom_priority).any():
        raise NotImplementedError("geom_priority parameter not supported.")

    # check joints
    q_width = {0: 7, 1: 4, 2: 1, 3: 1}
    non_free = np.concatenate([[j != 0] * q_width[j] for j in mj.jnt_type])
    # if mj.qpos0[non_free].any():
    #   raise NotImplementedError(
    #       'The `ref` attribute on joint types is not supported.')

    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_pos), key=lambda x: x[0]):
        position = np.array([p for _, p in group])
        if not (position == position[0]).all():
            raise RuntimeError("invalid joint stack: only one joint position allowed")

    # check dofs
    jnt_range = mj.jnt_range.copy()
    jnt_range[~(mj.jnt_limited == 1), :] = np.array([-np.inf, np.inf])
    for typ, limit, stiffness in zip(mj.jnt_type, jnt_range, mj.jnt_stiffness):
        if typ == 0:
            if stiffness > 0:
                raise RuntimeError("brax does not support stiffness for free joints")
        elif typ == 1:
            if np.any(~np.isinf(limit)):
                raise RuntimeError("brax does not support joint ranges for ball joints")
        elif typ in (2, 3):
            continue
        else:
            raise RuntimeError(f"invalid joint type: {typ}")

    for _, group in itertools.groupby(zip(mj.jnt_bodyid, mj.jnt_type), key=lambda x: x[0]):
        typs = [t for _, t in group]
        if len(typs) == 1 and typs[0] == 0:
            continue  # free
        elif 0 in typs:
            raise RuntimeError("invalid joint stack: cannot stack free joints")
        elif 1 in typs:
            raise NotImplementedError("ball joints not supported")

    # check collision geometries
    for i, typ in enumerate(mj.geom_type):
        mask = mj.geom_contype[i] | mj.geom_conaffinity[i] << 32
        if typ == 5:  # Cylinder
            _, halflength = mj.geom_size[i, 0:2]
            if halflength > 0.001 and mask > 0:
                raise NotImplementedError("Cylinders of half-length>0.001 are not supported for collision.")


def fuse_bodies(xml: str):
    """Fuses together parent child bodies that have no joint."""
    xml = ElementTree.fromstring(xml)
    _fuse_bodies(xml)
    return ElementTree.tostring(xml, encoding="unicode")


def load_mjmodel(path: Union[str, epath.Path]) -> mujoco.MjModel:
    """Loads an mj model from a MuJoCo mjcf file path."""
    elem = ElementTree.fromstring(epath.Path(path).read_text())
    _fuse_bodies(elem)
    meshdir = _get_meshdir(elem)
    assets = _find_assets(elem, epath.Path(path), meshdir)
    xml = ElementTree.tostring(elem, encoding="unicode")
    mj = mujoco.MjModel.from_xml_string(xml, assets=assets)
    return mj


class XMLVisualDataContainer:

    def __init__(self, xml_path: str):
        self.axes = []
        mj = load_mjmodel(xml_path)
        self.mj = mj

        self.body_mesh_dict = {}
        body_global_pos = mj.body_pos * 1
        body_global_quat = mj.body_quat * 1
        for i in range(body_global_pos.shape[0]):
            if i > 0:
                body_global_pos[i] += body_global_pos[mj.body_parentid[i]]
                mujoco.mju_mulQuat(
                    body_global_quat[i],
                    body_global_quat[mj.body_parentid[i]],
                    body_global_quat[i],
                )
        for i in range(mj.ngeom):
            my_pos = mj.geom_pos[i] * 0
            my_quat = body_global_quat[mj.geom_bodyid[i]] * 1
            mujoco.mju_mulQuat(my_quat, my_quat, mj.geom_quat[i])
            m = np.eye(3).reshape(-1)
            mujoco.mju_quat2Mat(m, my_quat)
            m = m.reshape(3, 3)
            if mj.geom_bodyid[i] not in self.body_mesh_dict:
                self.body_mesh_dict[mj.geom_bodyid[i]] = pv.PolyData()
            offset = mj.geom_pos[i]
            if mj.geom_type[i] == 2:
                mesh = pv.Sphere(radius=mj.geom_size[i][0] * 1, center=my_pos)
            elif mj.geom_type[i] == 3:
                mesh = pv.Capsule(
                    center=my_pos,
                    direction=(0, 0, 1),
                    cylinder_length=mj.geom_size[i][1] * 2,
                    radius=mj.geom_size[i][0],
                )
            elif mj.geom_type[i] == 6:  # box
                bounds = np.stack([-mj.geom_size[i], mj.geom_size[i]], axis=-1).reshape(-1)
                mesh = pv.Box(bounds)
            else:
                continue
            mesh.points = np.einsum("nj,ij->ni", mesh.points, m)
            mesh.points += offset
            self.body_mesh_dict[mj.geom_bodyid[i]] += mesh

            ax = pv.Axes()
            self.axes.append(ax)

        self.meshes = list(self.body_mesh_dict.values())[1:]
        # self.meshes = list(self.body_mesh_dict.values())


class NoteVisual:
    def __init__(self, pl):
        self.bloq_mesh = pv.read(f"{os.path.dirname(__file__)}/../assets/bloq.obj").scale(0.4)
        self.bloq_mesh.points -= self.bloq_mesh.center
        # self.bloq_mesh = pv.Cube(x_length=0.4, y_length=0.4, z_length=0.4)
        self.collider_mesh = pv.Cube(x_length=1.0, y_length=0.8, z_length=0.5)
        self.collider_mesh.points[..., 0] -= 0.25
        self.arrow_mesh = pv.read(f"{os.path.dirname(__file__)}/../assets/arrow.obj").scale(0.4).rotate_x(-90)
        self.arrow_mesh.points[..., 0] -= 0.4  # bring forward

        self.bloq_mesh.cell_data["color"] = np.array([[0, 0, 0]]).repeat(self.bloq_mesh.n_cells, 0)

        self.bloq_actor = pl.add_mesh(
            self.bloq_mesh,
            rgb=True,
            scalars="color",
        )
        self.collider_actor = pl.add_mesh(
            self.collider_mesh,
            color="red",
            style="wireframe",
        )
        self.arrow_actor = pl.add_mesh(self.arrow_mesh, color="white")
        self.bloq_actor.SetVisibility(False)
        self.arrow_actor.SetVisibility(False)
        self.collider_actor.SetVisibility(False)


class ObstacleVisual:
    def __init__(self, pl):
        self.collider_mesh = pv.Cube(x_length=0.4, y_length=0.4, z_length=0.4)
        self.collider_actor = pl.add_mesh(self.collider_mesh, color="green", style="wireframe")
        self.collider_actor.SetVisibility(False)


@dataclass
class GenericVisual:
    mesh: pv.PolyData
    actor: pv.Plotter

    def __post_init__(self):
        self.actor.SetVisibility(False)
