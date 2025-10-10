import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def plot_np2(channel_positions=None):
    num_shanks = 4
    tot_sites = 5120
    shank_pitch = 250
    shank_width = 70
    shank_length = 10000
    tip_length = 175
    site_size = (12, 12)
    horizontal_site_pitch = 32
    vertical_site_pitch = 15
    bank_size = 2700

    num_sites_per_shank = tot_sites // num_shanks
    if channel_positions is not None:
        # Round coordinates and convert to a set of tuples for very fast searching
        highlight_coords = set(map(tuple, np.round(channel_positions).astype(int)))
    else:
        highlight_coords = set()

    fig, ax = plt.subplots(figsize=(8, 12))
    for i in range(num_shanks):
        # DRAW SHANK
        shank_start_x = i * shank_pitch
        shank_color = "#bdc3c7"

        body = patches.Rectangle(
            (shank_start_x, tip_length),
            shank_width,
            shank_length - tip_length,
            facecolor=shank_color,
            edgecolor="black",
        )
        ax.add_patch(body)

        tip = patches.Polygon(
            [
                (shank_start_x, tip_length),
                (shank_start_x + shank_width, tip_length),
                (shank_start_x + shank_width / 2, 0),
            ],
            facecolor=shank_color,
            edgecolor="black",
        )
        ax.add_patch(tip)

        # DRAW RECORDING SITES
        shank_center_x = shank_start_x + shank_width / 2
        col_x_coords = [
            shank_center_x - horizontal_site_pitch / 2 + 8,
            shank_center_x + horizontal_site_pitch / 2 + 8,
        ]

        num_sites_per_column = num_sites_per_shank // 2

        for j in range(num_sites_per_column):
            # Calculate the y-position, same for both columns in a given row
            site_y = tip_length + (j * vertical_site_pitch)

            # Stop drawing if sites go beyond the shank's physical length
            if (site_y + site_size[1]) > shank_length:
                break

            # Draw site in the first column
            site1_x = col_x_coords[0] - site_size[0] / 2
            site1_center_x = col_x_coords[0]
            site1_center_y = site_y - tip_length
            site1_coords = (int(round(site1_center_x)), int(round(site1_center_y)))
            facecolor1 = "lime" if site1_coords in highlight_coords else "none"

            site1 = patches.Rectangle(
                (site1_x, site_y),
                site_size[0],
                site_size[1],
                facecolor=facecolor1,
                edgecolor="k",
                linewidth=0.2,
            )
            ax.add_patch(site1)

            # Draw site in the second column
            site2_x = col_x_coords[1] - site_size[0] / 2
            site2_center_x = col_x_coords[1]
            site2_center_y = site_y - tip_length
            site2_coords = (int(round(site2_center_x)), int(round(site2_center_y)))
            facecolor2 = "lime" if site2_coords in highlight_coords else "none"
            site2 = patches.Rectangle(
                (site2_x, site_y),
                site_size[0],
                site_size[1],
                facecolor=facecolor2,
                edgecolor="k",
                linewidth=0.2,
            )
            ax.add_patch(site2)

        # DRAW BANKS
        # y_start_sites = tip_length
        # num_banks = int(np.ceil((shank_length - y_start_sites) / bank_size))

        # for i in range(num_banks):
        #     bank_top_y = y_start_sites + ((i + 1) * bank_size)

        #     boundary_color = "#ff9393"
        #     if bank_top_y < shank_length:
        #         ax.axhline(
        #             y=bank_top_y, color=boundary_color, linestyle="--", linewidth=0.5
        #         )

    # 5. Finalize and style the plot
    ax.set_xlabel("Distance (µm)", fontsize=12)
    ax.set_ylabel("Distance from Tip (µm)", fontsize=12)
    ax.set_xlim(-shank_pitch / 2, (num_shanks - 0.5) * shank_pitch)
    ax.set_ylim(-tip_length, shank_length * 1.05)

    fig.tight_layout()
    # plt.show()


def plot_peak_heatmap(channel_positions, peak_channels):
    """
    Plots the NP 2.0 probe geometry with heatmap showing locations of neurons based on peak channel

    Args:
        channel_positions (np.ndarray): Array of shape (num_channels, 2) with [x, z] coordinates of channels.
        peak_channels (np.ndarray): 1D array of channel indices for each neuron's peak.
    """
    unique_channels, counts = np.unique(peak_channels, return_counts=True)
    peak_counts = dict(zip(unique_channels, counts))
    max_count = max(peak_counts.values())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0, vmax=max_count)

    num_shanks = 4
    shank_pitch = 250
    shank_width = 70
    shank_length = 10000
    tip_length = 175
    site_size = (12, 12)

    # 4. Setup plot
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_aspect("equal")
    ax.set_title("Neuron Peak Channel Density", fontsize=16, pad=20)

    # 5. Draw shanks
    for i in range(num_shanks):
        shank_start_x = i * shank_pitch
        shank_color = "#bdc3c7"
        # Draw body and tip
        body = patches.Rectangle(
            (shank_start_x, tip_length),
            shank_width,
            shank_length - tip_length,
            facecolor=shank_color,
            edgecolor="black",
            zorder=1,
        )
        ax.add_patch(body)
        tip = patches.Polygon(
            [
                (shank_start_x, tip_length),
                (shank_start_x + shank_width, tip_length),
                (shank_start_x + shank_width / 2, 0),
            ],
            facecolor=shank_color,
            edgecolor="black",
            zorder=1,
        )
        ax.add_patch(tip)

    # 6. Draw sites as a heatmap
    for channel_id, (x_pos, z_pos) in enumerate(channel_positions):
        count = peak_counts.get(channel_id, 0)

        # Determine color and style based on count
        if count > 0:
            face_color = cmap(norm(count))
            edge_color = "k"
            line_width = 0.2
        else:
            face_color = "none"
            edge_color = "#d3d3d3"  # Faint grey for unused sites
            line_width = 0.1

        # The channel_positions gives the center, so calculate the bottom-left corner
        bottom_left_x = x_pos - site_size[0] / 2
        bottom_left_y = z_pos - site_size[1] / 2

        site = patches.Rectangle(
            (bottom_left_x, bottom_left_y),
            site_size[0],
            site_size[1],
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=line_width,
            zorder=2,  # Draw sites on top of the shank
        )
        ax.add_patch(site)

    # 7. Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30, shrink=0.8)
    cbar.set_label("Number of Neurons", rotation=270, labelpad=15, fontsize=12)

    # 8. Finalize and style the plot
    ax.set_xlabel("Distance (µm)", fontsize=12)
    ax.set_ylabel("Depth (µm)", fontsize=12)
    ax.set_xlim(-shank_pitch / 2, (num_shanks - 0.5) * shank_pitch)
    ax.set_ylim(-tip_length, shank_length * 1.05)
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()
