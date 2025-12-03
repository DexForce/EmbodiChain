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
import platform
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.patches import Circle, Rectangle
from matplotlib import colors as mcolors
from embodichain.utils.logger import log_error
from typing import Dict, List
from operator import sub

from matplotlib import rc

x_min, x_max = 0.275, 1.125
y_min, y_max = -0.425, 0.425
bins = 100


def draw_keypoints(
    rgb: np.ndarray, keypoints_2d: np.ndarray, color_dict: dict = None
) -> np.ndarray:
    import cv2

    keypoints_2d = np.nan_to_num(keypoints_2d, nan=0)
    assert (
        keypoints_2d.max(0)[0] <= 1 and keypoints_2d.max(0)[1] <= 1
    ), keypoints_2d.max(0)
    assert (
        keypoints_2d.min(0)[0] >= 0 and keypoints_2d.min(0)[1] >= 0
    ), keypoints_2d.min(0)
    n = keypoints_2d.shape[0]
    color = [(255 - i / n * 255, 0, i / n * 255) for i in range(n)]
    height, width = rgb.shape[:2]

    rgb = np.copy(rgb)

    for i in range(n):
        assigned_color = False
        if color_dict is not None:
            for key_ids, color_str in color_dict.items():
                if i in key_ids:
                    color[i] = tuple(
                        int(chl * 255) for chl in mcolors.to_rgb(color_str)[::-1]
                    )
                    assigned_color = True
                    break
            if not assigned_color:
                log_error(
                    f"Once color_dict is provided, all the keypoints ought to be colored, but got {i} not colored."
                )

        # Draw the keypoint
        rgb = cv2.circle(
            rgb.copy(),
            (int(keypoints_2d[i][0] * width), int(keypoints_2d[i][1] * height)),
            2,
            color[i],
            2,
        )

    return rgb


def draw_action_distribution(
    actions: Dict[str, np.ndarray],
    indices: Dict[str, List[int]] = None,
    output_path: str = None,
    smooth: bool = False,
    return_data: bool = False,
):
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    key_names = indices.keys() if indices is not None else actions.keys()
    data = {}
    for key_name in key_names:
        qpos = (
            actions[
                :,
                indices[key_name],
            ]
            if indices is not None
            else actions[key_name]
        )
        num_dim = qpos.shape[1]
        min_square = int(np.ceil(np.sqrt(num_dim)))
        rowcol = (min_square, min_square)

        fig, axs = plt.subplots(rowcol[0], rowcol[1], figsize=(20, 20))
        for i in range(num_dim):
            row = i // rowcol[0]
            col = i % rowcol[1]
            ax_i = axs[row, col] if min_square != 1 else axs
            ax_i.plot(
                (
                    qpos[:, i]
                    if not smooth
                    else gaussian_filter1d(qpos[:, i], sigma=3, axis=0, mode="nearest")
                ),
                marker="o",
                ms=2,
            )
            ax_i.set_title(f"{key_name}_{i}")

        plt.tight_layout()
        data[key_name] = fig
        if output_path is not None and os.path.exists(output_path):
            plt.savefig(
                os.path.join(output_path, "action_distribution_{}.png".format(key_name))
            )

    if return_data:
        return data


def draw_feature(
    feature_list: List[np.ndarray], vis_images: List[np.ndarray]
) -> List[np.ndarray]:
    import cv2
    from copy import deepcopy

    vis_features = []
    for feature, image in zip(feature_list, vis_images):
        feature_ = cv2.resize(
            feature,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        image = cv2.addWeighted(deepcopy(image), 0.5, feature_, 0.5, 0)
        vis_features.append(image)
    return vis_features


class HeatMapEnv:
    def __init__(self, is_success):
        """Initialize the drawing environment and static elements"""
        self.points = []
        self.b_fail_points = []
        self.c_fail_points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_aspect("equal")

        circle1 = Circle(
            (0.7, 0),
            radius=0.425,
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
            label="Circle Zone",
        )
        circle2 = Circle(
            (0.233, 0.3),
            radius=0.08,
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
            label="Circle Zone",
        )
        circle3 = Circle(
            (0.233, -0.3),
            radius=0.08,
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
            label="Circle Zone",
        )

        rectangle1 = Rectangle(
            (0.67, -0.22),
            0.16,
            0.16,
            angle=0,
            fill=False,
            edgecolor="blue",
            linewidth=2,
            linestyle="-.",
            label="Rect Zone",
        )

        rectangle2 = Rectangle(
            (0.67, 0.06),
            0.16,
            0.16,
            angle=0,
            fill=False,
            edgecolor="green",
            linewidth=2,
            linestyle="-.",
            label="Rect Zone",
        )

        for patch in [circle1, circle2, circle3, rectangle1, rectangle2]:
            self.ax.add_patch(patch)

        self.ax.set(
            xlim=(x_min, x_max),
            ylim=(y_min, y_max),
            xticks=np.arange(x_min, x_max + 0.01, 0.04),
            yticks=np.arange(y_min, y_max + 0.01, 0.04),
        )
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.set_title("Real-time Heatmap")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        hist = np.zeros((bins, bins))
        self.im = self.ax.imshow(
            hist.T,
            origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="Greys",
            norm=LogNorm(vmin=0.1, vmax=10),
        )

        self.cbar = self.fig.colorbar(self.im)
        self.cbar.set_label("Density")
        self.is_success = is_success
        if self.is_success:
            text = "Success_Points_Pair: 0"
        else:
            text = "Bottle_Fail_Points: 0\nCup_Fail_Points: 0"
        self.text_label = self.ax.text(
            0.95,
            0.95,
            text,
            transform=self.ax.transAxes,
            fontsize=14,
            color="red",
            ha="right",
        )

        plt.ion()
        plt.show(block=False)
        plt.tight_layout()

    def update_heatmap(self, new_point, new_fail):
        if self.is_success:
            self.points.append(new_point)
            x_coords = [p[0] for p in self.points]
            y_coords = [p[1] for p in self.points]
        else:
            if new_fail == 0:
                self.b_fail_points.append(new_point)
                x_coords = [p[0] for p in self.b_fail_points]
                y_coords = [p[1] for p in self.b_fail_points]
            else:
                self.c_fail_points.append(new_point)
                x_coords = [p[0] for p in self.c_fail_points]
                y_coords = [p[1] for p in self.c_fail_points]

        hist, x_edges, y_edges = np.histogram2d(
            x_coords, y_coords, bins=bins, range=[[x_min, x_max], [y_min, y_max]]
        )

        self.im.set_data(hist.T)

        if self.is_success:
            self.text_label.set_text(f"Success_Points_Pair: {len(self.points)/2}")
        else:
            if new_fail == 0:
                self.text_label.set_text(
                    f"Bottle_Fail_Points: {len(self.b_fail_points)}\nCup_Fail_Points: {len(self.c_fail_points)}"
                )
            else:
                self.text_label.set_text(
                    f"Bottle_Fail_Points: {len(self.b_fail_points)}\nCup_Fail_Points: {len(self.c_fail_points)}"
                )
        # im.autoscale()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def save_map(self):
        if self.is_success:
            plt.savefig("./outputs/success_heatmap.png")
        else:
            plt.savefig("./outputs/fail_heatmap.png")


# TeX support: on Linux assume TeX in /usr/bin, on OSX check for texlive
if (platform.system() == "Darwin") and "tex" in os.getenv("PATH"):
    LATEX = True
elif (platform.system() == "Linux") and os.path.isfile("/usr/bin/latex"):
    LATEX = True
else:
    LATEX = False

# setup pyplot w/ tex support
if LATEX:
    rc("text", usetex=True)


class Package:
    """Encapsulation of a work package

    A work package is instantiated from a dictionary. It **has to have**
    a label, astart and an end. Optionally it may contain milestones
    and a color

    :arg str pkg: dictionary w/ package data name
    """

    def __init__(self, pkg):

        DEFCOLOR = "#32AEE0"

        self.label = pkg["label"]
        self.start = pkg["start"]
        self.end = pkg["end"]

        if self.start < 0 or self.end < 0:
            raise ValueError("Package cannot begin at t < 0")
        if self.start > self.end:
            raise ValueError("Cannot end before started")

        try:
            self.milestones = pkg["milestones"]
        except KeyError:
            pass

        try:
            self.color = pkg["color"]
        except KeyError:
            self.color = DEFCOLOR

        try:
            self.legend = pkg["legend"]
        except KeyError:
            self.legend = None


# https://github.com/stefanSchinkel/gantt/tree/master
class Gantt:
    """Gantt
    Class to render a simple Gantt chart, with optional milestones
    """

    def __init__(self, dict: Dict):
        """Instantiation

        Create a new Gantt using the data in the file provided
        or the sample data that came along with the script

        :arg str dataFile: file holding Gantt data
        """

        # some lists needed
        self.packages = []
        self.labels = []

        self._loadData(dict)
        self._procData()

    def _loadData(self, data):
        """Load data from a JSON file that has to have the keys:
        packages & title. Packages is an array of objects with
        a label, start and end property and optional milesstones
        and color specs.
        """

        # must-haves
        self.title = data["title"]

        for pkg in data["packages"]:
            self.packages.append(Package(pkg))

        self.labels = [pkg["label"] for pkg in data["packages"]]

        # optionals
        self.milestones = {}
        for pkg in self.packages:
            try:
                self.milestones[pkg.label] = pkg.milestones
            except AttributeError:
                pass

        try:
            self.xlabel = data["xlabel"]
        except KeyError:
            self.xlabel = ""
        try:
            self.xticks = data["xticks"]
        except KeyError:
            self.xticks = ""

    def _procData(self):
        """Process data to have all values needed for plotting"""
        # parameters for bars
        self.nPackages = len(self.labels)
        self.start = [None] * self.nPackages
        self.end = [None] * self.nPackages

        for pkg in self.packages:
            idx = self.labels.index(pkg.label)
            self.start[idx] = pkg.start
            self.end[idx] = pkg.end

        self.durations = map(sub, self.end, self.start)
        self.yPos = np.arange(self.nPackages, 0, -1)

    def format(self):
        """Format various aspect of the plot, such as labels,ticks, BBox
        :todo: Refactor to use a settings object
        """
        # format axis
        plt.tick_params(
            axis="both",  # format x and y
            which="both",  # major and minor ticks affected
            bottom="on",  # bottom edge ticks are on
            top="off",  # top, left and right edge ticks are off
            left="off",
            right="off",
        )

        # tighten axis but give a little room from bar height
        plt.xlim(0, max(self.end))
        plt.ylim(0.5, self.nPackages + 0.5)

        # add title and package names
        plt.yticks(self.yPos, [label.replace("qpos", "") for label in self.labels])
        plt.title(self.title)

        if self.xlabel:
            plt.xlabel(self.xlabel)

        if self.xticks:
            plt.xticks(self.xticks, map(str, self.xticks))

    def add_milestones(self):
        """Add milestones to GANTT chart.
        The milestones are simple yellow diamonds
        """

        if not self.milestones:
            return

        x = []
        y = []
        for key in self.milestones.keys():
            for value in self.milestones[key]:
                y += [self.yPos[self.labels.index(key)]]
                x += [value]

        plt.scatter(
            x, y, s=120, marker="D", color="yellow", edgecolor="black", zorder=3
        )

    def add_legend(self):
        """Add a legend to the plot iff there are legend entries in
        the package definitions
        """
        cnt = 0
        legends = []
        for pkg in self.packages:
            if pkg.legend not in legends:
                cnt += 1
                idx = self.labels.index(pkg.label)
                self.barlist[idx].set_label(pkg.legend)
                legends.append(pkg.legend)

        if cnt > 0:
            self.legend = self.ax.legend(shadow=False, ncol=3, fontsize="medium")

    def render(self):
        """Prepare data for plotting"""

        # init figure
        self.fig, self.ax = plt.subplots()
        self.ax.yaxis.grid(False)
        self.ax.xaxis.grid(True)

        # assemble colors
        colors = []
        for pkg in self.packages:
            colors.append(pkg.color)

        self.barlist = plt.barh(
            self.yPos,
            list(self.durations),
            left=self.start,
            align="center",
            height=0.5,
            alpha=1,
            color=colors,
        )

        # format plot
        self.format()
        self.add_milestones()
        self.add_legend()

    @staticmethod
    def show():
        """Show the plot"""
        plt.show()

    @staticmethod
    def save(saveFile="img/GANTT.png"):
        """Save the plot to a file. It defaults to `img/GANTT.png`.

        :arg str saveFile: file to save to
        """
        plt.savefig(saveFile, bbox_inches="tight")


def visualize_trajectory(poses: np.ndarray):
    """Visualizes a 3D trajectory and its z-axis directions.

    This function takes a series of 4x4 transformation matrices representing
    poses in 3D space and visualizes the trajectory along with the z-axis
    directions at each pose.

    Args:
        poses (np.ndarray): A numpy array of shape (N, 4, 4), where N is the
            number of poses. Each pose is a 4x4 transformation matrix.

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # positions of the trajectory
    positions = poses[:, :3, 3]
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        "r-",
        linewidth=3,
        label="轨迹",
    )

    # direction of z-axis
    for i in range(len(poses)):
        R = poses[i, :3, :3]
        t = poses[i, :3, 3]

        z_axis = R[:, 2] * 0.01
        ax.quiver(
            t[0],
            t[1],
            t[2],
            z_axis[0],
            z_axis[1],
            z_axis[2],
            color="blue",
            arrow_length_ratio=0.2,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.show()
