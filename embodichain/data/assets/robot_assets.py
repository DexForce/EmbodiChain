# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------


import os
import open3d as o3d
from embodichain.data.dataset import EmbodiChainDataset
from embodichain.data.constants import (
    EMBODICHAIN_DOWNLOAD_PREFIX,
    EMBODICHAIN_DEFAULT_DATA_ROOT,
)


robot_assets = "robot_assets"


class CobotMagicArm(EmbodiChainDataset):
    """Dataset class for the Cobot Magic Arm robot.

    Reference:
        https://global.agilex.ai/products/cobot-magic

    Directory structure:
        CobotMagicArm/
            CobotMagicNoGripper.urdf
            CobotMagicWithGripperV70.urdf
            CobotMagicWithGripperV70NewUV.urdf
            CobotMagicWithGripperV70NoMaterial.urdf
            CobotMagicWithGripperV100.urdf
            CobotMagicWithGripperV100NewUV.urdf
            CobotMagicWithGripperV100NoMaterial.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import CobotMagicArm
        >>> dataset = CobotMagicArm()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "CobotMagicArmV2.zip"
            ),
            "14af3e84b74193680899a59fc74e8337",
        )
        prefix = "CobotMagicArm"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class RidgeBack(EmbodiChainDataset):
    """Dataset class for the RidgeBack wheeled robot.

    Reference:
        https://clearpathrobotics.com/ridgeback-indoor-robot-platform/

    Directory structure:
        RidgeBack/
            RidgeBack.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import RidgeBack
        >>> dataset = RidgeBack()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("RidgeBack/RidgeBack.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "RidgeBack.zip"),
            "f03e1a6f4c781ad8957a88bdb010e9b6",
        )
        prefix = "RidgeBack"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class UnitreeH1(EmbodiChainDataset):
    """Dataset class for the Unitree H1 robot.

    Reference:
        https://www.unitree.com/h1/

    Directory structure:
        UnitreeH1/
            UnitreeH1.urdf
            UnitreeH1WithWrist.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import UnitreeH1
        >>> dataset = UnitreeH1()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("UnitreeH1/UnitreeH1.urdf"))
        >>> print(get_data_path("UnitreeH1/UnitreeH1WithWrist.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "UnitreeH1.zip"),
            "339417cef5051a912693f3c64d29dddc",
        )
        prefix = "UnitreeH1"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class ABB(EmbodiChainDataset):
    """Dataset class for the ABB robot.

    Reference:
        https://global.abb/

    Directory structure:
        ABB/
            IRB1200_5_90/IRB1200_5_90.urdf
            IRB2600_12_165/IRB2600_12_165.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import ABB
        >>> dataset = ABB()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("ABB/IRB1200_5_90/IRB1200_5_90.urdf"))
        >>> print(get_data_path("ABB/IRB2600_12_165/IRB2600_12_165.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "ABB.zip"),
            "ea6df4983982606c43387783e5fb8c05",
        )
        prefix = "ABB"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Motoman(EmbodiChainDataset):
    """Dataset class for the Motoman robot.

    Reference:
        https://www.motoman.com/en-us

    Directory structure:
        Motoman/
            GP7/GP7.urdf
            GP12/GP12.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Motoman
        >>> dataset = Motoman()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Motoman/GP7/GP7.urdf"))
        >>> print(get_data_path("Motoman/GP12/GP12.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "Motoman.zip"),
            "ee5f16cfce34d8e2cb996fcff8a25986",
        )
        prefix = "Motoman"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class KUKA(EmbodiChainDataset):
    """Dataset class for the KUKA robot.

    Reference:
        https://www.kuka.com/

    Directory structure:
        KUKA/
            KUKA/KR6_R700_sixx/KR6_R700_sixx.urdf
            KUKA/KR6_R900_sixx/KR6_R900_sixx.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import ABB
        >>> dataset = ABB()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("KUKA/KR6_R700_sixx/KR6_R700_sixx.urdf"))
        >>> print(get_data_path("KUKA/KR6_R900_sixx/KR6_R900_sixx.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "KUKA.zip"),
            "da7a2dfd0db3f486e407f038d25c7537",
        )
        prefix = "KUKA"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Fanuc(EmbodiChainDataset):
    """Dataset class for the Fanuc robot.

    Reference:
        https://www.fanuc.com/

    Directory structure:
        Fanuc/
            M_20iA/M_20iA.urdf
            R_2000iC_165F/R_2000iC_165F.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Fanuc
        >>> dataset = Fanuc()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Fanuc/KR6_R700_sixx/KR6_R700_sixx.urdf"))
        >>> print(get_data_path("Fanuc/KR6_R900_sixx/KR6_R900_sixx.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "Fanuc.zip"),
            "0a1c562f4719f7cdc1b24545fec4a301",
        )
        prefix = "Fanuc"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class UniversalRobots(EmbodiChainDataset):
    """Dataset class for the Universal Robots.

    Reference:
        https://www.universal-robots.com/products/ur-series/

    Directory structure:
        UniversalRobots/
            UR3/UR3.urdf
            UR3e/UR3e.urdf
            UR5/UR5.urdf
            UR5e/UR5e.urdf
            UR10/UR10.urdf
            UR10e/UR10e.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import UniversalRobots
        >>> dataset = UniversalRobots()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("UniversalRobots/UR3/UR3.urdf"))
        >>> print(get_data_path("UniversalRobots/UR3e/UR3e.urdf"))
        >>> print(get_data_path("UniversalRobots/UR5/UR5.urdf"))
        >>> print(get_data_path("UniversalRobots/UR5e/UR5e.urdf"))
        >>> print(get_data_path("UniversalRobots/UR10/UR10.urdf"))
        >>> print(get_data_path("UniversalRobots/UR10e/UR10e.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(
                EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "UniversalRobots.zip"
            ),
            "dbd12f7e36cef4e5025b82f748233b80",
        )
        prefix = "UniversalRobots"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Rokae(EmbodiChainDataset):
    """Dataset class for the Rokae robots.

    Reference:
        https://www.rokae.com/en/product/show/349/SR-Cobots.html

    Directory structure:
        Rokae/
            SR3/SR3.urdf
            SR5/SR5.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Rokae
        >>> dataset = Rokae()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Rokae/SR3/SR3.urdf"))
        >>> print(get_data_path("Rokae/SR5/SR5.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "Rokae.zip"),
            "fbfb852d6139e94b7c422771542f988f",
        )
        prefix = "Rokae"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Franka(EmbodiChainDataset):
    """Dataset class for the Franka robots.

    Reference:
        https://franka.de/franka-research-3

    Directory structure:
        Franka/
            Panda/Panda.urdf
            PandaHand/PandaHand.urdf
            PandaWithHand/PandaWithHand.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Franka
        >>> dataset = Franka()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Franka/Panda/Panda.urdf"))
        >>> print(get_data_path("Franka/PandaHand/PandaHand.urdf"))
        >>> print(get_data_path("Franka/PandaWithHand/PandaWithHand.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "Franka.zip"),
            "c2de367fe1da02eeb45a8129f903d0b6",
        )
        prefix = "Franka"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Agile(EmbodiChainDataset):
    """Dataset class for the Agile robots.

    Reference:
        https://www.agile-robots.com/en/solutions/diana-7/

    Directory structure:
        Agile/
            Diana7/Diana7.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Agile
        >>> dataset = Agile()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Agile/Diana7/Diana7.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "Agile.zip"),
            "fd47d7ab8a4d13960fd76e59544ba836",
        )
        prefix = "Agile"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Hans(EmbodiChainDataset):
    """Dataset class for the Hans robots.

    Reference:
        https://www.huayan-robotics.com/elfin

    Directory structure:
        Hans/
            E05/E05.urdf
            E10/E10.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Hans
        >>> dataset = Hans()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Hans/E05/E05.urdf"))
        >>> print(get_data_path("Hans/E10/E10.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "Hans.zip"),
            "c867c406e3dffd6982fd0a15e7dc7e29",
        )
        prefix = "Hans"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class Aubo(EmbodiChainDataset):
    """Dataset class for the Aubo robots.

    Reference:
        https://www.aubo-robotics.cn/

    Directory structure:
        Aubo/
            i5/i5.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import Aubo
        >>> dataset = Aubo()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("Aubo/i5/i5.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "Aubo.zip"),
            "2574649cd199c11267cc0f4aeac65557",
        )
        prefix = "Aubo"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)


class RainbowY1(EmbodiChainDataset):
    """Dataset class for the Aubo robots.

    Reference:
        https://www.rainbow-robotics.com/en_rby1

    Directory structure:
        RainbowY1/
            RainbowY1.urdf

    Example usage:
        >>> from embodichain.data.robot_dataset import RainbowY1
        >>> dataset = RainbowY1()
        or
        >>> from embodichain.data import get_data_path
        >>> print(get_data_path("RainbowY1/RainbowY1.urdf"))
    """

    def __init__(self, data_root: str = None):
        data_descriptor = o3d.data.DataDescriptor(
            os.path.join(EMBODICHAIN_DOWNLOAD_PREFIX, robot_assets, "RainbowY1.zip"),
            "5979a3aaadb5de6488b13765d523564f",
        )
        prefix = "RainbowY1"
        path = EMBODICHAIN_DEFAULT_DATA_ROOT if data_root is None else data_root

        super().__init__(prefix, data_descriptor, path)
