from controller import Camera, Lidar, Supervisor
import numpy as np
import matplotlib.pyplot as plt
# Pytorch dependensies
import torch
from torch import multiprocessing

from collections import defaultdict

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (Compose, DoubleToFloat, ObservationNorm, StepCounter,
                          TransformedEnv)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from vehicle import Driver
from controller import Camera, Lidar, Supervisor
import numpy as np

# Initializing vehicle values
speed = 0
angle = 0
maxSpeed = 1.8 # Altino's maximum speed
maxLeft = -3.14
maxRight = 3.14

altino.setSteeringAngle(angle)
altino.setCruisingSpeed(speed)
headlightsOn = True


# For PPO:
# Make a PPO with only Lidar input, avoiding obstacles during continuous driving

# Minimal discrete action space
actions = [
    (-0.5, 1.2),   # left
    (0.0, 1.2),    # straight
    (0.5, 1.2),    # right
    (0.0, 0.6),    # slow
    (0.0, 0.0)     # stop
]
# Larger discrete action space
# steering: [-0.6, -0.3, 0, 0.3, 0.6]
# speed:    [0.6, 1.2]


# Creating an environment
class WebotsEnv:

    def __init__(self):
        # Setting up Altino robot
        self.altino = vehicle()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.headlights = self.altino.getDevice("headlights")
        self.backlights = self.altino.getDevice("backlights")

        # Setting up sensors
        self.lidar = self.robot.getDevice("lidar")
        #self.camera = self.robot.getDevice("camera")

        self.lidar.enable(self.timestep)
        #self.camera.enable(self.timestep)


    def reset(self, altino):
        # reset robot position using supervisor
        altino.setCustomData(0)

        observation = self.get_observation()
        return observation, {}

    def step(self, action):

        self.apply_action(action)

        self.robot.step(self.timestep)

        observation = self.get_observation()
        reward = self.compute_reward()

        terminated = self.check_collision()
        truncated = False

        return observation, reward, terminated, truncated, {}





while altino.step() != -1:



    if fCenterVal > 400 and fCenterVal < 600:
        speed -= (0.01 * speed)
    elif fCenterVal > 600 and fCenterVal < 800:
        speed /= 1.01
    if backVal > 400 and backVal < 600:
        speed /= 1.01
    elif backVal > 600 and backVal < 800:
        speed /= 1.1

    if fLeftVal > fRightVal:
        angle += (fLeftVal - fRightVal) / (300 * sensorMax)
    elif fRightVal > fLeftVal:
        angle -= (fRightVal - fLeftVal) / (300 * sensorMax)
    else:
        angle /= 1.5

    if sLeftVal > 300:
        angle += 0.003
    if sRightVal > 300:
        angle -= 0.003

    speed += 0.001

# clamp speed and angle to max values
if speed > maxSpeed:
    speed = maxSpeed
elif speed < -1 * maxSpeed:
    speed = -1 * maxSpeed
if angle > 0.4:
    angle = 0.4
elif angle < -0.4:
    angle = -0.4
    

# SENSOR PROCESSING
# image = camera.getImage()
# width = camera.getWidth()
# height = camera.getHeight()
# image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
# rgb = image_array[:, :, :3]
# camera.saveImage("image.png", 1000)

range_array = lidar.getRangeImage()
res = lidar.getHorizontalResolution()
fov = lidar.getFov()




#if (printCounter % 10) == 0:
    #print("Welcome to the Altino Sample Controller")
    #print("----------------------------------------------")
    #print("This sample controller is based on a Braitenberg vehicle, \n")
    #print("it uses the vehicle's infrared distance sensors to avoid obstacles.")
    #print("\n-----------------Controls---------------------")
    #print("'M' to enable manual control")
    #print("'N' to disable manual control")
    #print("'H' to turn on the headlights")
    #print("'G' to turn off the headlights")
    #print("Arrow Keys to accelerate, decelerate and turn")
    #print("Space bar to brake (manual mode only)")
    #print("----------------------------------------------")
    #print("Current Wheel Angle and Throttle values:")
    #print("Angle: %.2f" % angle)
    #print("Throttle: %.1f " % (100 * speed / maxSpeed))
    #if useManual:
    #    print("---------Manual Control ENABLED-------------")
    #else:
    #    print("---------Manual Control DISABLED------------")
altino.setCruisingSpeed(speed)
altino.setSteeringAngle(angle)

