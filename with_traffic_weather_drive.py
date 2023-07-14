
#Imports
import glob
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
     
except IndexError:
    pass



#Used for weather
#Used for route planning. Path will have to be changed to where your directory is.
#adding to create a gps route from start to finish
sys.path.append('C:\CARLA_0.9.14\WindowsNoEditor\PythonAPI\carla')
from agents.navigation.global_route_planner import GlobalRoutePlanner

import math
import random
import numpy as np
from matplotlib import cm
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
import carla
from keras.models import load_model
#Select portions of code
#inspired by sentdex


#Global Sensor Settings. Can be adjusted to your needs.
#Sensors in the simEnv below are set to use the global
#variable settings

#Set image size for carla version. May need to be adjusted
#depending on version of carla. X and Y sizes work for 9.14
IMAGE_SIZE_X = 640
IMAGE_SIZE_Y = 360
RGB_FOV = 110
RADAR_HORIZONTAL_FOV = 30
RADAR_VERTICAL_FOV = 30
RADAR_POINTS_PER_SECOND = 5500
SAMPLING_RESOLUTION = 4.5 #used for route planning


#AGENT SETTINGS
FPS = 60
SECONDS_PER_EPISODE = 120
REPLAY_MEMORY_SIZE = 1000
MODEL_NAME = "Xception"
MODEL_PATH = None
BATCH_SIZE = 10
MIN_REWARD = -650
EPISODES = 150
DISCOUNT = .97
MAX_EPSILON = 1
EPSILON_DECAY = .995
EPSILON_MIN = .01
GAMMA = .99
AGGREGATE_STATS_EVERY = 3
NUM_ACTIONS = 4
IMAGE_INPUT_SHAPE = (IMAGE_SIZE_Y, IMAGE_SIZE_X, 3)   
POINT_CLOUD_INPUT_SHAPE = (RADAR_POINTS_PER_SECOND,4)
SPEED_INPUT_SHAPE = (1,)
GPS_INPUT_SHAPE = (3,)
FUTURE_GPS_INPUT_SHAPE = (3,)
WEATHER_SPEED_FACTOR = .5


#wil be used to write data to log files at the end of each run
log_distance_from_start_list = []
log_distance_to_end_list = []
log_vehicle_speed_list = []
log_ep_rewards = []
log_route_distance_list = []
log_loss_list = []

#Weather classes taken from CARLA EXAMPLE directory
def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._t = clamp(delta + self._t, -250.0, 100.0)
        self.clouds = clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._t + delay, 0.0, 85.0)
        self.wetness = clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        self.fog = clamp(self._t - 10, 0.0, 30.0)
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._storm.fog
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)

#Classes to manage the simulator
class simEnv:
    
    #Env variables
    #Set display sizes for images. Will need to be adjusted
    #Based on your version of carla
    im_width = IMAGE_SIZE_X
    im_height = IMAGE_SIZE_Y
    
    #In case of user input to kill simulation, destroy
    #all assets in the sim
    def destroy_all_assets(self):
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
    
    
    #Setup object
    def __init__(self):
        #Connect to the client
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(8.0) #Quit after 8 seconds of no response

        
        #Get the world we are in from the client
        self.world = self.client.get_world()
        
        #will hold cars in the simulator
        self.vehicle_list = []
        

        self.speed_factor = WEATHER_SPEED_FACTOR

        
        #Collect blueprints to spawn in actors
        self.bp_lib = self.world.get_blueprint_library()
        
        #Get spawn points for this map --> map might need to be global to edit this on reset
        self.spawn_points = self.world.get_map().get_spawn_points()
        
        #Set our main car, will need to be edited later to setup
        #correct camera positions. 
        self.micro_bp = self.bp_lib.find('vehicle.micro.microlino')
        
        #Will hold data from each sensor
        
        self.sensor_data = {
                        'left_rgb_image'  : np.zeros((self.im_height, self.im_width,4)),
                        'right_rgb_image' : np.zeros((self.im_height, self.im_width,4)),
                        'front_rgb_image' : np.zeros((self.im_height, self.im_width,4)),
                        'front_radar_data': tf.Variable(tf.zeros(POINT_CLOUD_INPUT_SHAPE)),
                        'back_radar_data' : tf.Variable(tf.zeros(POINT_CLOUD_INPUT_SHAPE)),
                        'imu_data'        : {
                                            'gyro': carla.Vector3D(),
                                            'accel': carla.Vector3D(),
                                            'compass': 0
                        },
                        'gnss_data'       : tf.constant([0.0,0.0,0.0],dtype=tf.float32),
                        'carla_xyz_loc'   : carla.Location(),
                        'collision'       : False
        }
        
        
        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._min_speed = 10
        self.turn_speed = 15
        self._sampling_resolution = SAMPLING_RESOLUTION
        self.MAX_STEER = 1
        self.start_time = None
        
        #Handle command line arguments
        self.render_op,self.weather_op,self.traffic_op = self.check_cmd_args(sys.argv)

        #Set render
        self.set_render_option(self.render_op)
        #set weather
        self.weather_control = self.set_weather_option(self.weather_op)
        #set traffic
        self.vehicle_list = self.set_traffic_option(self.traffic_op)
        
        


    def check_cmd_args(self, args):
    
        #assume defalt settings
        render = 'DEFAULT'
        weather = 'DEFAULT'
        traffic = 'DEFAULT'
        


        if len(args) == 4:    #Command line arguments
            render = args[1]
            weather = args[2]
            traffic = args[3]

        elif len(args) != 1:
            print('There are 2 command line choices.')
            print('1. Pass no arguments. for default behavior')
            print('\tThis is no rendering, sunny weather, medium traffic (20 cars)')
            print('2. Pass in ALL command line arguments.')
            print('Your arguments:', sys.argv, 'len ==', len(sys.argv))
            print('Render Option should be sent as first command line argument.')
            print('You may enter T, R, or RENDER to set the server to render mode.')
            print('You may enter F, NR, NORENDER, DEF, DEFAULT to set the server to NO render mode.')
            print()
            print('Weather should be sent as the second command line argument.')
            print('You may enter ON which will rotate weather from dark, light, rainy and foggy.')
            print('You may enter DEF or DEFAULT or OFF which will keep the weather sunny.')
            print('')
            print('Traffic Option should be the third command line argument.')
            print('For Small traffic (10 Cars) enter S, SM, SMALL')
            print('For Medium traffic (20 cars) enter M, MD, MED, MEDIUM, DEF, DEFAULT.')
            print('For Large traffic (30 cars) enter L, LG, LARGE')
            print('You May also pass in any number you want eg. 12, 200, 17 if you want')
            print()
            
            exit()
            
        
        return render,weather,traffic
            
        
    #Processing functions for different sensors
    #Callback functions store the data from a sensor and process it.
    #Most are setup to store data into a dictionary
    #Which can then be accessed at a later point
    def lrgb_callback(self, image, ldata_dict):
        im = np.array(image.raw_data)
        im2 = im.reshape((self.im_height, self.im_width,4))
        final_im = im2[:,:,:3]
        ldata_dict['left_rgb_image'] = final_im
        #ldata_dict['left_rgb_image'] = np.reshape(np.copy(image.raw_data), (image.width, image.height, 4))

    def rrgb_callback(self, image, rdata_dict):
        im = np.array(image.raw_data)
        im2 = im.reshape((self.im_height, self.im_width,4))
        final_im = im2[:,:,:3]
        rdata_dict['right_rgb_image'] = final_im
        #rdata_dict['right_rgb_image'] = np.reshape(np.copy(image.raw_data), (image.width, image.height, 4))

    def frgb_callback(self, image, fdata_dict):
        im = np.array(image.raw_data)
        im2 = im.reshape((self.im_height, self.im_width,4))
        final_im = im2[:,:,:3]
        fdata_dict['front_rgb_image'] = final_im
        #fdata_dict['front_rgb_image'] = np.reshape(np.copy(image.raw_data), (image.width, image.height, 4))

    def gnss_callback(self, data, data_dict):
        lat_rad = (np.deg2rad(data.latitude) + np.pi) % (2 * np.pi) - np.pi
        lon_rad = (np.deg2rad(data.longitude) + np.pi) % (2 * np.pi) - np.pi
        R = 6378135 # Aequatorradii
        x = R * np.sin(lon_rad) * np.cos(lat_rad)       # iO
        y = R * np.sin(-lat_rad)  
        z = data.altitude

        # store as carla location for buiolt in carla function
        data_dict['carla_xyz_loc'] = carla.Location(x,y,z)

        location_as_tensor = tf.constant([x,y,z], shape=(3,), dtype=tf.float32)
        
        data_dict['gnss_data'] = location_as_tensor

    def imu_callback(self, data, data_dict):
        data_dict['imu_data'] = {
            'gyro': data.gyroscope,
            'accel': data.accelerometer,
            'compass': data.compass
        }

    def collision_callback(self, event, data_dict):
        data_dict['collision'] = True

    def fradar_callback(self, data, data_dict):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (-1, 4))
        tensor_points = np.zeros(POINT_CLOUD_INPUT_SHAPE)
        tensor_points[:len(points)] = points
        data_dict['front_radar_data'] = tensor_points
    
    def bradar_callback(self, data, data_dict):
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (-1, 4))
        tensor_points = np.zeros(POINT_CLOUD_INPUT_SHAPE)
        tensor_points[:len(points)] = points
        data_dict['back_radar_data'] = tensor_points
        
    #Core component of environment    
    #Setup start and end locations for this simulation, make a route between them.
    #Spawn in actors and sensors. Set sensors listening
    def reset(self):
    
        #Set start and end spots for this simulation
        #use .location to get just x,y,z. As is these come with pitch and yaw from Rotation class
        self.start_location = random.choice(self.spawn_points)
        self.end_location = random.choice(self.spawn_points)
        #ex: start_xyz = start_location.location

        #Record route distance
        routeDist = self.start_location.location.distance(self.end_location.location)
        log_route_distance_list.append(routeDist)
        
        #Collect a route from start to finish on this map.
        glob_route_plan = GlobalRoutePlanner(self.world.get_map(), self._sampling_resolution)
        
        #Trace out the route into a set of waypoints / steps to get from start to end. 
        self.sim_route = glob_route_plan.trace_route(self.start_location.location, self.end_location.location)

        
        #Spawn in the vehicle at the start location
        self.vehicle = self.world.try_spawn_actor(self.micro_bp, self.start_location)
        #If path is blocked by a car
        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.micro_bp, self.start_location)
        if self.render_op.upper() in ('T', 'R', 'RENDER', 'ON', 'O'):
            spectator = self.world.get_spectator()
            transform = carla.Transform(self.vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),
                                        self.vehicle.get_transform().rotation) 
            spectator.set_transform(transform)
        
        #@@@@@@@@@@@
        #SETUP SENSORS
        #Cameras
        #Then Radars
        #Then IMU
        #Lastly, GNSS
        #@@@@@@@@@@@
        
        #Get sensor specs from global variables
        cam_fov = RGB_FOV
        radar_horiz_fov = RADAR_HORIZONTAL_FOV
        radar_vert_fov = RADAR_VERTICAL_FOV
        radar_points_per_second = RADAR_POINTS_PER_SECOND
    
        #Transform Positions to attach cameras on the **microlino** car
        #If you are not using the microlino, these transforms WILL have to be ADJUSTED
        left_cam_x =.50
        left_cam_y = -.65
        left_cam_z = .80
        left_cam_yaw = -155

        right_cam_x = .50
        right_cam_y = .80
        right_cam_z = 1.0
        right_cam_yaw = 155
        
        front_cam_x = 1.0
        front_cam_z = .85
        

        #Transform Positions to attach the radars on the microlino car
        #Positioning is centered on the car so y is typically not a necessary cooridante. If you need to 
        #offset the camera left or right, you will need to add in a y coordinate
        #If you are not using the microlino, these transforms will have to be adjusted
        front_radar_x = 1.0
        front_radar_z = .85

        back_radar_x = -1.0
        back_radar_z = .85
        
        
        #Get rgb camera BP and set settings
        self.rgb_cam_bp = self.bp_lib.find('sensor.camera.rgb')
        self.rgb_cam_bp.set_attribute("image_size_x", str(self.im_width)) 
        self.rgb_cam_bp.set_attribute("image_size_y", str(self.im_height)) 
        self.rgb_cam_bp.set_attribute("fov", str(cam_fov)) 
        
        #Setup left camera, attach to the car
        left_cam_transform = carla.Transform(carla.Location(x=left_cam_x,y=left_cam_y,z=left_cam_z), carla.Rotation(yaw=left_cam_yaw))
        self.left_cam = self.world.spawn_actor(self.rgb_cam_bp, left_cam_transform, attach_to = self.vehicle)
        
        #Setup right camera, attach to the car
        right_cam_transform = carla.Transform(carla.Location(x=right_cam_x,y=right_cam_y,z=right_cam_z), carla.Rotation(yaw=right_cam_yaw))
        self.right_cam = self.world.spawn_actor(self.rgb_cam_bp, right_cam_transform, attach_to = self.vehicle)
        
        #Setup front camrea, attach to the car
        front_cam_transform = carla.Transform(carla.Location(x=front_cam_x, z = front_cam_z))
        self.front_cam = self.world.spawn_actor(self.rgb_cam_bp, front_cam_transform, attach_to = self.vehicle)
        
        
        #Get radar BP and set settings
        self.radar_bp = self.bp_lib.find('sensor.other.radar')
        self.radar_bp.set_attribute('horizontal_fov', str(radar_horiz_fov))
        self.radar_bp.set_attribute('vertical_fov', str(radar_vert_fov))
        self.radar_bp.set_attribute('points_per_second', str(radar_points_per_second))
        
        #Setup and attach front radar to car
        front_radar_transform = carla.Transform(carla.Location(x=front_radar_x, z=front_radar_z))
        self.front_radar = self.world.spawn_actor(self.radar_bp, front_radar_transform, attach_to=self.vehicle)
        
        #Setup and attach back radar to car
        back_radar_transform = carla.Transform(carla.Location(x=back_radar_x, z=back_radar_z))
        self.back_radar = self.world.spawn_actor(self.radar_bp, back_radar_transform, attach_to=self.vehicle)
        
        
        #Setup IMU - Inertial Measurement Unit
        self.imu_bp = self.bp_lib.find('sensor.other.imu')
        self.imu = self.world.spawn_actor(self.imu_bp, carla.Transform(), attach_to = self.vehicle)
        
        #Setup GPS - GNSS is global navigation satellite system
        #GPS is USA specific, using GNSS to be consistent with sim. 
        self.gnss_bp = self.bp_lib.find('sensor.other.gnss')
        self.gnss = self.world.spawn_actor(self.gnss_bp, carla.Transform(), attach_to = self.vehicle)
        
        #The car drops into the simulator. To avoid registering a collision on spawn in, sleep for 3 second
        #So the car can land
        time.sleep(2)
        
        #Setup collision sensor so we can penalize the car if it hits something.
        self.collision_bp = self.bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to = self.vehicle)
        
       
        
        
        #Set sensors listening.
        #listening determines the behavior of each sensor as the world ticks. 
        
        #listen for each camera --> Might have to do workaround to display.
        self.left_cam.listen(lambda image:  self.rrgb_callback(image, self.sensor_data))
        self.right_cam.listen(lambda image: self.lrgb_callback(image, self.sensor_data))
        self.front_cam.listen(lambda image: self.frgb_callback(image, self.sensor_data))
        
        
        
        #listen for each radar
        self.front_radar.listen(lambda data: self.fradar_callback(data, self.sensor_data))
        self.back_radar.listen(lambda data: self.bradar_callback(data, self.sensor_data))
        
        #listen for imu
        self.imu.listen(lambda event: self.imu_callback(event, self.sensor_data))
        
        #listen for gnss
        self.gnss.listen(lambda event: self.gnss_callback(event, self.sensor_data))
        
        #listen for colission
        self.collision_sensor.listen(lambda event: self.collision_callback(event, self.sensor_data))

        
        
        #Supposedly helps with startup time
        self.vehicle.apply_control(carla.VehicleControl(throttle=.0,brake=0.0))
        
        
    
    #Send the current environment readings.
    #returns an array of image data and an array of radar data, then data from the imu and gnss
    def get_state(self):
        elapsed_time = self.start_time - self.get_time()
        
        #Tick weather 
        if self.weather_control is not None:
            self.weather_control.tick(self.speed_factor * elapsed_time)
            self.world.set_weather(self.weather_control.weather)
        
        #Get current env data and tore in array
        image_data = []
        radar_data = []
        image_data.append(self.sensor_data['left_rgb_image'])
        image_data.append(self.sensor_data['right_rgb_image'])
        image_data.append(self.sensor_data['front_rgb_image'])
        radar_data.append(self.sensor_data['front_radar_data'])
        radar_data.append(self.sensor_data['back_radar_data'])
        cur_gps_waypoint = self.sensor_data['gnss_data'] #[x,y,z]
        future_gps_waypoint = self.get_next_waypoint(self.sensor_data['carla_xyz_loc'])
        
        
        #Get current speed
        self._speed = self.get_speed()
        #turn into tensor for processing
        speed_tensor = tf.constant([self._speed], shape=SPEED_INPUT_SHAPE, dtype=tf.float32)

        #Record speed of vehicle
        log_vehicle_speed_list.append(self._speed)
       
        return image_data, radar_data, speed_tensor, cur_gps_waypoint, future_gps_waypoint
    
    #Takes as input the current location, and returns the location of the next waypoint (xyz loc)
    #in the gps route to the finish
    def get_next_waypoint(self, cur_location):
        
        nextWaypoint = None
        minDist = 100000
        #distances = []
        #for waypoint in self.sim_route:
        #    distances.append(cur_location.distance(waypoint[0].transform.location))
        #minDist = min(distances)
        #next_waypoint_idx = distances.index(minDist)
        #next_waypoint = self.sim_route[next_waypoint_idx]
        #loop through the sim route to find waypoint the car is closest to now
        for x, waypoint in enumerate(self.sim_route):
            
        
            curDist = cur_location.distance(waypoint[0].transform.location)
            if curDist < minDist:
                #set closest waypoint
                nextWaypoint = waypoint
                minDist = curDist
        #nextWaypoint is the waypoint the car is closest to. This is typically a 
        #waypoint the car has just travelled over. Update next waypoint to be a 
        #waypoint (5 meters away)
        next_waypoint = nextWaypoint[0].next(5)[0].transform.location

        #convert to tensor 
        tensor_waypoint = tf.constant([next_waypoint.x,next_waypoint.y,next_waypoint.z], shape=(3,), dtype=tf.float32)
        return tensor_waypoint
            
        
    #At each time step what should the agent be doing
    def step(self, action):
        #Currently 5 supported vehicle output actions
        #0 == stay straight
        #1 == turn left
        #2 == turn right
        #3 == brake

        #Get speed limit of the area.
        self._speed_limit = self.vehicle.get_speed_limit()

        #Get the vehicle to perform an action
        #Continue driving straight
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle = self._speed_limit, steer = 0))
            
        #Turn to the left
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle = self.turn_speed, steer = -1*self.MAX_STEER))
            
        #Turn to the right
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle = self.turn_speed, steer = 1*self.MAX_STEER))
            
        #Brake
        else: #action == 3:
            self.vehicle.apply_control(carla.VehicleControl(brake = 1))
            
            
        #Determine next state of vehicle
        next_image_inputs, next_point_cloud_inputs, next_speed_input, next_cur_gps_input, \
        next_future_gps_input = self.get_state()
        
        
        done = False
        reward = 0
        #Apply some rewards / penalties
        if self.sensor_data['collision'] == True:
            #We have hit something. Stop, -250 points
            done = True
            
            reward = -250
            #Currently penalizing all collisions the same. This should change in the future
            
        #If we are going slow, add a slight penalty. 
        elif self._speed < self._min_speed:
            reward = -2
            
        #If we arrived at the end location
        elif self.vehicle.get_transform() == self.end_location:
            done = True
            reward = 200
            print('MADE IT!')
            
            
        reward += self.apply_dist_to_end_reward(self.vehicle.get_transform().location, self.end_location.location)
        #left off returning sensors. Might need to do that.     
        return next_image_inputs, next_point_cloud_inputs, next_speed_input, next_cur_gps_input, \
            next_future_gps_input, reward, done
        
        
    #Set the penalty based on how far the car is from the end point
    def apply_dist_to_end_reward(self,loc1, loc2):
        
        reward = 0
        
        #calculate distance from p1 to p2
        distance = loc1.distance(loc2)
        
        #apply penalty based on distance
        if distance in range(0,6):
            reward = 50
            print("SUPER CLOSE")
        elif distance in range(6,16):
            reward = 15
            print("Sort of Close")
        elif distance in range(16,31):
            reward = -20
        elif distance in range(31-46):
            reward = -30
        else:
            reward = -50
        
        return reward

    #Stop and dDestroy the sensors and vehicles used in the simulation
    #Also resets the collision tracker to False
    def cleanup(self):
        self.right_cam.stop()
        self.left_cam.stop()
        self.front_cam.stop()
        self.front_radar.stop()
        self.back_radar.stop()
        self.gnss.stop()
        self.imu.stop()
        self.collision_sensor.stop()

        #reset collision tracker
        self.sensor_data['collision'] = False

        
        self.right_cam.destroy()
        self.left_cam.destroy()
        self.front_cam.destroy()
        self.front_radar.destroy()
        self.back_radar.destroy()
        self.gnss.destroy()
        self.imu.destroy()
        self.collision_sensor.destroy()
        self.vehicle.destroy()

        #Get rid of other traffic

        #for v in self.vehicle_list:
        #    if v is not None and v.destroy() is False:
        #        v.destroy()

        #reset vehicle list for next simulation
        #self.vehicle_list = self.vehicle_list.clear()

    #Will return the distance the car is from the start 
    #and then the end location
    def get_distances(self):
        cur_loc = self.vehicle.get_transform().location

        dist_from_start = self.start_location.location.distance(cur_loc)

        dist_to_end = self.end_location.location.distance(cur_loc)

        return dist_from_start, dist_to_end
    def get_speed(self):
        velocity = self.vehicle.get_velocity()
        speed = int(3.6 * math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2))
        return speed

    
    #Set render settings on the simulator
    def set_render_option(self, op):

        option = op.upper()
        self.settings = self.world.get_settings()
        
        if option in ('T', 'R', 'RENDER', 'ON'):
            #SET RENDER MODE
            self.settings.no_rendering_mode = False
            
        elif option in ('F', 'NR', 'NORENDER', 'DEF', 'DEFAULT') : #SET NO RENDER MODE
            
            self.settings.no_rendering_mode = True
        else:
            print('Render option unrecognized. Got', op)
            print('Render Option should be sent as first command line argument.')
            print('You may enter T, R, or RENDER to set the server to render mode.')
            print('You may enter F, NR, NORENDER, DEF, DEFAULT to set the server to NO render mode.')
        self.world.apply_settings(self.settings)

    def set_weather_option(self, op):
        #Do Nothing, stay sunny
        weather = None
        
        #Set weather to rotate
        if op.upper() == 'ON':
            weather = Weather(self.world.get_weather())
        
        return weather
        
    #Determine how many cars to spawn
    def set_traffic_option(self, op):

        if op.isdigit():
            num_cars = int(op)
        elif op.upper() not in('M', 'MD', 'MED', 'MEDIUM', 'DEF', 'DEFAULT','S', 'SM', 'SMALL','L', 'LG', 'LARGE'):
            print('Car traffic option not recognized. Got', op)
            print('Traffic Option should be the third command line argument.')
            print('For Small traffic (10 Cars) enter S, SM, SMALL')
            print('For Medium traffic (20 cars) enter M, MD, MED, MEDIUM, DEF, DEFAULT.')
            print('For Large traffic (30 cars) enter L, LG, LARGE')
            print('You May also pass in any number you want eg. 12, 200, 17 if you want')
        else:
            option = op.upper()
            if option in ('M', 'MD', 'MED', 'MEDIUM', 'DEF', 'DEFAULT'):
                num_cars = 20
            elif option in ('S', 'SM', 'SMALL'):
                num_cars = 10
            elif option in ('L', 'LG', 'LARGE'):
                num_cars = 30


        
        car_list = []
        #Try to spawn in vehicles. Add to actor List.
        for v in range(num_cars):
            vehicle_bp = random.choice(self.bp_lib.filter('vehicle'))
            npc = self.world.try_spawn_actor(vehicle_bp, random.choice(self.spawn_points))
            if npc is not None:
                car_list.append(npc)
        
        
        for car in car_list:
            
            car.set_autopilot(True)

        
        return car_list    
        
            

    def get_time(self):
        return time.time()

        
        
class DQNAgent():
    def __init__(self, image_input_shape,point_cloud_input_shape, speed_input_shape, gps_input_shape, future_gps_input_shape, num_actions):
        
        #Set input shapes and num actions possible
        self.image_input_shape = image_input_shape
        self.point_cloud_input_shape = point_cloud_input_shape
        self.speed_input_shape = speed_input_shape
        self.gps_input_shape = gps_input_shape
        self.future_gps_input_shape = future_gps_input_shape
        self.num_actions=num_actions
        self.epsilon = MAX_EPSILON
        self.gamma = GAMMA
        
        #Set replay buffer
        self.replay_buffer = deque(maxlen=REPLAY_MEMORY_SIZE)
        
        #Setup q and target network
        self.q_network, self.trainable_variables = self.create_model()
        self.target_network, _ = self.create_model()
        self.target_network.set_weights(self.q_network.get_weights())
        
        

        #Define optimizer and discount
        self.optimizer = tf.keras.optimizers.Adam()
        self.discount_factor = DISCOUNT

        
        #print(self.target_network.summary())
        
        #If we are loading weights from a pretrained model, use those. 
        if MODEL_PATH is not None:
            self.load_weights(MODEL_PATH)
    
    
    #Load weights from presaevd model
    def load_weights(self, path):
        weights = np.load(path)
        self.set_weights(weights)

    def set_weights(self, weights):
        
        self.q_network.set_weights(weights)
        self.target_network.set_weights(weights)

    #Define mode architecture    
    def create_model(self):

      
        # Image inputs
        image_input_1 = layers.Input(shape=self.image_input_shape)
        image_input_2 = layers.Input(shape=self.image_input_shape)
        image_input_3 = layers.Input(shape=self.image_input_shape)
        
        # Xception model for image processing
        image_model = Xception(weights='imagenet', include_top=False, input_shape=self.image_input_shape)
        image_features_1 = image_model(image_input_1)
        image_features_2 = image_model(image_input_2)
        image_features_3 = image_model(image_input_3)
        image_features_1_flat = layers.Flatten()(image_features_1)
        image_features_2_flat = layers.Flatten()(image_features_2)
        image_features_3_flat = layers.Flatten()(image_features_3)

        
        # Point cloud inputs
        point_cloud_input_1 = layers.Input(shape=self.point_cloud_input_shape)
        point_cloud_input_2 = layers.Input(shape=self.point_cloud_input_shape)
        point_cloud_features_1 = layers.Flatten()(point_cloud_input_1)
        point_cloud_features_2 = layers.Flatten()(point_cloud_input_2)

        # Speed input
        speed_input = layers.Input(shape=(1,))

        # GPS inputs
        gps_input = layers.Input(shape=self.gps_input_shape)
        future_gps_input = layers.Input(shape=self.future_gps_input_shape)

        # Concatenate inputs
        concatenated_inputs = layers.concatenate([
            image_features_1_flat, image_features_2_flat, image_features_3_flat,
            point_cloud_features_1, point_cloud_features_2,
            speed_input, gps_input, future_gps_input
        ], axis=1)

       

        # Hidden layers
        hidden_layer_1 = layers.Dense(512, activation='relu')(concatenated_inputs)
        hidden_layer_2 = layers.Dense(256, activation='relu')(hidden_layer_1)
        hidden_layer_3 = layers.Dense(128, activation='relu')(hidden_layer_2)

        # Output layer
        output_layer = layers.Dense(self.num_actions)(hidden_layer_3)
        

        # Create the model
        model = tf.keras.Model(inputs=[image_input_1, image_input_2, image_input_3,
                                    point_cloud_input_1, point_cloud_input_2,
                                    speed_input, gps_input, future_gps_input], outputs=output_layer)

        trainable_variables = model.trainable_variables

        # Compile the model
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

        
        return model, trainable_variables
    
        
    #Update the data in our replay memory for training
    def update_replay_memory(self, image_inputs, point_cloud_inputs, speed_input, gps_input, future_gps_input, action, 
                             reward, next_image_inputs, next_point_cloud_inputs, next_speed_input, next_gps_input, 
                             next_future_gps_input, done):

    
        self.replay_buffer.append((image_inputs, point_cloud_inputs, speed_input, gps_input, future_gps_input, action, 
                                    reward, next_image_inputs, next_point_cloud_inputs, next_speed_input, next_gps_input,
                                    next_future_gps_input, done))
        
    def update_q_network(self):     
        #if there are not enough samples, do nothing
        if(len(self.replay_buffer) < BATCH_SIZE):
            return
 

        batch = random.sample(self.replay_buffer, BATCH_SIZE)
        image_inputs, point_cloud_inputs, speed_inputs, gps_inputs, future_gps_inputs, \
        actions, rewards, next_image_inputs, next_point_cloud_inputs, next_speed_inputs, \
        next_gps_inputs, next_future_gps_inputs, dones = zip(*batch)

        

        #seperate out individual iamges and radar cloud
        image_input_1 = image_inputs[0]
        image_input_2 = image_inputs[1]
        image_input_3 = image_inputs[2]
        point_cloud_input_1 = point_cloud_inputs[0]
        point_cloud_input_2 = point_cloud_inputs[1]

        next_image_input_1 = next_image_inputs[0]
        next_image_input_2 = next_image_inputs[1]
        next_image_input_3 = next_image_inputs[2]
        next_point_cloud_input_1 = next_point_cloud_inputs[0]
        next_point_cloud_input_2 = next_point_cloud_inputs[1]

        

        # Convert the lists to numpy arrays
        image_input_1 = np.array(image_input_1)
        image_input_2 = np.array(image_input_2)
        image_input_3 = np.array(image_input_3)
        
        speed_inputs = np.array(speed_inputs)
        gps_inputs = np.array(gps_inputs)
        future_gps_inputs = np.array(future_gps_inputs)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_image_input_1 = np.array(next_image_input_1)
        next_image_input_2 = np.array(next_image_input_2)
        next_image_input_3 = np.array(next_image_input_3)
        next_point_cloud_input_1 = np.array(next_point_cloud_input_1)
        next_point_cloud_input_2 = np.array(next_point_cloud_input_2)
        next_speed_inputs = np.array(next_speed_inputs)
        next_gps_inputs = np.array(next_gps_inputs)
        next_future_gps_inputs = np.array(next_future_gps_inputs)
        dones = np.array(dones)

        
        
        #Reshape All inputs so that 1st dimension is None to pass into the model
           
        reshape_image_input_1 = np.reshape(image_input_1, (-1,IMAGE_SIZE_Y, IMAGE_SIZE_X, 3))
        reshape_image_input_2 = np.reshape(image_input_2, (-1,IMAGE_SIZE_Y, IMAGE_SIZE_X, 3))
        reshape_image_input_3 = np.reshape(image_input_3, (-1,IMAGE_SIZE_Y, IMAGE_SIZE_X, 3))

        reshape_image_input_1 = reshape_image_input_1[:2]
        reshape_image_input_2 = reshape_image_input_2[:2]
        reshape_image_input_3 = reshape_image_input_3[:2]

        reshape_point_cloud_input_1 = np.reshape(point_cloud_input_1, (-1, RADAR_POINTS_PER_SECOND,4))
        reshape_point_cloud_input_2 = np.reshape(point_cloud_input_2, (-1, RADAR_POINTS_PER_SECOND,4))

        reshape_point_cloud_input_1 = reshape_point_cloud_input_1[:2] 
        reshape_point_cloud_input_2 = reshape_point_cloud_input_2[:2]
        
        reshape_speed_inputs = np.reshape(speed_inputs, (-1,1,))
        reshape_gps_inputs = np.reshape(gps_inputs, (-1,3,))
        reshape_future_gps_inputs = np.reshape(future_gps_inputs, (-1,3))
        
        reshape_speed_inputs = reshape_speed_inputs[:2]
        reshape_gps_inputs = reshape_gps_inputs[:2]
        reshape_future_gps_inputs = reshape_future_gps_inputs[:2]
        
        #nexts inputs reshaping
        reshape_next_image_input_1 = np.reshape(image_input_1, (-1,IMAGE_SIZE_Y, IMAGE_SIZE_X, 3))
        reshape_next_image_input_2 = np.reshape(image_input_2, (-1,IMAGE_SIZE_Y, IMAGE_SIZE_X, 3))
        reshape_next_image_input_3 = np.reshape(image_input_3, (-1,IMAGE_SIZE_Y, IMAGE_SIZE_X, 3))

        reshape_next_image_input_1 = reshape_next_image_input_1[:2]
        reshape_next_image_input_2 = reshape_next_image_input_2[:2]
        reshape_next_image_input_3 = reshape_next_image_input_3[:2]

        reshape_next_point_cloud_input_1 = np.reshape(next_point_cloud_input_1, (-1, RADAR_POINTS_PER_SECOND,4))
        reshape_next_point_cloud_input_2 = np.reshape(next_point_cloud_input_1, (-1, RADAR_POINTS_PER_SECOND,4))

        reshape_next_point_cloud_input_1 = reshape_next_point_cloud_input_1[:2]
        reshape_next_point_cloud_input_2 = reshape_next_point_cloud_input_2[:2]

        reshape_next_speed_inputs = np.reshape(next_speed_inputs, (-1,1,))
        reshape_next_gps_inputs = np.reshape(next_gps_inputs, (-1,3,))
        reshape_next_future_gps_inputs = np.reshape(next_future_gps_inputs, (-1,3,))

        reshape_next_speed_inputs = reshape_next_speed_inputs[:2]
        reshape_next_gps_inputs = reshape_next_gps_inputs[:2]
        reshape_next_future_gps_inputs = reshape_next_future_gps_inputs[:2]

        dones = dones[:2]
        rewards = rewards[:2]
        actions = actions[:2]

        # Epsilon-greedy policy for action selection
        next_q_values = self.target_network.predict([[reshape_next_image_input_1,reshape_next_image_input_2,reshape_next_image_input_3], \
                [reshape_next_point_cloud_input_1,reshape_next_point_cloud_input_2], reshape_next_speed_inputs, reshape_next_gps_inputs, \
                reshape_next_future_gps_inputs])

        max_next_q_values = np.max(next_q_values, axis=1)

        
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
    
        # Train the q_network using the target q_values
        with tf.GradientTape() as tape:
            q_values = self.q_network([[reshape_image_input_1,reshape_image_input_2,reshape_image_input_3], [reshape_point_cloud_input_1,reshape_point_cloud_input_2],\
                                    reshape_speed_inputs, reshape_gps_inputs, reshape_future_gps_inputs])
            q_values_actions = tf.reduce_sum(q_values * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.keras.losses.MSE(target_q_values, q_values_actions)
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        log_loss_list.append(loss.numpy())

        

        


    #determine what action to take. 
    def get_action(self, image_inputs, point_cloud_inputs, speed_input, gps_input, future_gps_input):
        # Epsilon-greedy policy for action selection
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.num_actions)
            time.sleep(1/FPS)
            return action
        else:
            image_inputs = [np.expand_dims(image, axis=0) for image in image_inputs]
            point_cloud_inputs = [np.expand_dims(pc_input, axis=0) for pc_input in point_cloud_inputs]
            speed_input = np.reshape(speed_input, (1, 1))
            gps_input_reshaped = np.reshape(gps_input, (1, 3))
            future_gps_input_reshaped = np.reshape(future_gps_input, (1, 3))
            q_values = self.q_network([*image_inputs, *point_cloud_inputs, speed_input, gps_input_reshaped, future_gps_input_reshaped])
            return np.argmax(q_values[0])
                                        
                                       
                                       
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
        time.sleep(2.5) #getting weights can take time and cause conflicts with other
        #code sleeping here to allow for more time to get weights and updates
    
    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY
            
def main():

    global MIN_REWARD

    if not os.path.isdir('models'):
        os.makedirs('models')
        
    #So results are more easily repeated
    random.seed(1)
    tf.random.set_seed(1)


    #Get sim env up and running, andmake an agent
    carla_simulator = simEnv()

    agent = DQNAgent(IMAGE_INPUT_SHAPE, POINT_CLOUD_INPUT_SHAPE, SPEED_INPUT_SHAPE, GPS_INPUT_SHAPE, FUTURE_GPS_INPUT_SHAPE, NUM_ACTIONS)
    

    # Define training parameters
    num_episodes = 30
    max_steps_per_episode = 30
    episode_duration = SECONDS_PER_EPISODE

    #Arrays to keep track of stats as we go
    
    ep_dist_to_next_waypoint = []
    
    
    
    #setup episode timeing
    ep_duration = SECONDS_PER_EPISODE
    carla_simulator.start_time = carla_simulator.get_time()

    # Training loop
    for episode in range(num_episodes):
        # Reset the environment and get initial state (starting environment)
        carla_simulator.reset() 
        image_inputs, point_cloud_inputs, speed_input, cur_gps_input, future_gps_input = carla_simulator.get_state() 

        episode_reward = 0  # Track the cumulative reward for the episode
        
        
        
        for step in range(max_steps_per_episode):
            
            

            # Choose an action
            action = agent.get_action(image_inputs, point_cloud_inputs, speed_input, cur_gps_input, future_gps_input)

            # Perform the action in the environment
            next_image_inputs, next_point_cloud_inputs, next_speed_input, next_cur_gps_input, \
                next_future_gps_input, reward, done = carla_simulator.step(action)

            


            #next_info = [next_image_inputs[0], next_image_inputs[1], next_image_inputs[2], next_point_cloud_inputs[0],next_point_cloud_inputs[1],\
            # next_speed_input, next_cur_gps_input, next_future_gps_input]
            
            if step % 5 == 0:
                # Update the replay buffer
                agent.update_replay_memory(image_inputs, point_cloud_inputs, speed_input, cur_gps_input, future_gps_input, action, 
                                 reward, next_image_inputs, next_point_cloud_inputs, next_speed_input, next_cur_gps_input, 
                                 next_future_gps_input, done)



            # Update the current state
            image_inputs, point_cloud_inputs, speed_input, cur_gps_input, future_gps_input = next_image_inputs,  \
                    next_point_cloud_inputs, next_speed_input, next_cur_gps_input, next_future_gps_input

            # Update the Q-network
            agent.update_q_network()

            # Update the target network periodically
            if step % AGGREGATE_STATS_EVERY == 0:
                agent.update_target_network()

            # Accumulate the episode reward
            episode_reward += reward

            

            
            time_passed = carla_simulator.get_time() - carla_simulator.start_time
            # Check if the episode is done, or over time limit
            if done or (time_passed > ep_duration):
                break


        

        

        # Decay epsilon
        agent.decay_epsilon()
        log_ep_rewards.append(episode_reward)
        
        if (episode % AGGREGATE_STATS_EVERY == 0) or (episode == 1):
            average_reward = sum(log_ep_rewards[-AGGREGATE_STATS_EVERY:])/len(log_ep_rewards[-AGGREGATE_STATS_EVERY:])
            local_min_reward = min(log_ep_rewards[-AGGREGATE_STATS_EVERY:])
            local_max_reward = max(log_ep_rewards[-AGGREGATE_STATS_EVERY:])

            local_dist_from_start, local_dist_to_dest = carla_simulator.get_distances()

            #add data to lists to write into files at the end of the program
            log_distance_from_start_list.append(local_dist_from_start)
            log_distance_to_end_list.append(local_dist_to_dest)
            

            # Save model, but only when min reward is greater or equal a set value
            if local_min_reward >= MIN_REWARD:
                agent.q_network.save(f'models/{MODEL_NAME}__{local_max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{local_min_reward:_>7.2f}min__{int(time.time())}.model')
                #Move up minimum acceptable reward
                MIN_REWARD+=50
        
        #Cleanup simulation for the next run
        carla_simulator.cleanup()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        carla_simulator = simEnv()
        
        print('User Interrupt. Destroying assets')
        carla_simulator.destroy_all_assets()
        pass
    except Exception as e:
        print('Training Exception.')
        print('EXCEPTION =', e)
        print('Passing for logging')
        pass
    finally:
        print('Simulation ended. Logging Files.')
        #Open Logging Files                                                   
        rewards_file = open('log_rewards.txt', 'a')                           
        dist_start_file = open('log_distance_from_start.txt', 'a')            
        dist_end_file = open('log_distance_to_end.txt', 'a')                  
        speed_file = open('log_car_speed_kph.txt', 'a') 
        route_length_file = open('log_route_distance.txt', 'a')
        loss_file = open('log_loss.txt', 'a')
        

        #Log rewards
        for x in log_ep_rewards:
            rewards_file.write(str(x)+'\n')

        #log dist from start
        for x in log_distance_from_start_list:
            dist_start_file.write(str(x)+'\n')

        #log dist to end
        for x in log_distance_to_end_list:
            dist_end_file.write(str(x)+'\n')

        #log speed
        for x in log_vehicle_speed_list:
            speed_file.write(str(x)+'\n')

        #Log route distance
        for x in log_route_distance_list:
            route_length_file.write(str(x)+'\n')

        for x in log_loss_list:
            loss_file.write(str(x)+'\n')

        

        print('Closing Files')
        rewards_file.close()
        dist_start_file.close()
        dist_end_file.close()
        speed_file.close()
        loss_file.close()
        

        carla_simulator = simEnv()
        
        print('Program Finished. Destroying assets')
        carla_simulator.destroy_all_assets()
        

