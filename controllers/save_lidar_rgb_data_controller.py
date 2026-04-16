"""Simple Webots recorder: RGB + LiDAR for 10 seconds."""

import csv
import json
import os
from typing import Tuple

import numpy as np
from controller import Camera, Lidar, Robot
from PIL import Image

RECORD_SECONDS = 10
OUTPUT_DIR = "recordings"


def make_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def record_rgb_lidar(robot: Robot, duration_sec: float, output_dir: str) -> None:
    """Record RGB frames and LiDAR scans for the requested duration."""
    timestep = int(robot.getBasicTimeStep())
    steps = int(np.ceil(duration_sec * 1000.0 / timestep))

    camera = robot.getDevice("camera")
    lidar = robot.getDevice("lidar")
    camera.enable(timestep)
    lidar.enable(timestep)

    width = camera.getWidth()
    height = camera.getHeight()
    lidar_resolution = lidar.getHorizontalResolution()
    lidar_max_range = lidar.getMaxRange()

    make_output_dir(output_dir)
    metadata = {
        "duration_seconds": duration_sec,
        "timestep_ms": timestep,
        "steps": steps,
        "camera_width": width,
        "camera_height": height,
        "lidar_resolution": lidar_resolution,
        "lidar_max_range": lidar_max_range,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=2)

    print(f"[RECORDER] Recording {steps} frames ({duration_sec}s) to '{output_dir}'")

    lidar_data = []  # Collect all lidar ranges for CSV

    for frame_idx in range(steps):
        if robot.step(timestep) == -1:
            print("[RECORDER] Simulation stopped early.")
            break

        image_bytes = camera.getImage()
        lidar_ranges = np.array(lidar.getRangeImage(), dtype=np.float32)

        # Save RGB as JPEG
        image_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 4))
        rgb = image_array[:, :, :3]
        rgb_image = Image.fromarray(rgb)
        rgb_path = os.path.join(output_dir, f"rgb_{frame_idx:04d}.jpg")
        rgb_image.save(rgb_path, 'JPEG')

        # Collect LiDAR data
        lidar_data.append([frame_idx] + lidar_ranges.tolist())

        if frame_idx % 20 == 0:
            print(f"[RECORDER] Saved frame {frame_idx + 1}/{steps}")

    # Save all LiDAR data to CSV
    lidar_csv_path = os.path.join(output_dir, "lidar_data.csv")
    with open(lidar_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        header = ['frame_idx'] + [f'range_{i}' for i in range(lidar_resolution)]
        writer.writerow(header)
        # Write data
        writer.writerows(lidar_data)

    print(f"[RECORDER] Finished recording to '{output_dir}'")


if __name__ == "__main__":
    robot = Robot()
    wheels = [
        robot.getDevice("left_front_wheel"),
        robot.getDevice("right_front_wheel"),
        robot.getDevice("left_rear_wheel"),
        robot.getDevice("right_rear_wheel"),
    ]

    for motor in wheels:
        motor.setPosition(float('inf'))  # Set to velocity control mode
        motor.setVelocity(27.8)  # Start with zero velocity
    record_rgb_lidar(robot, RECORD_SECONDS, OUTPUT_DIR)
