#!/usr/bin/env python

import glob
import os
import sys
import numpy as np
import cv2
try:
    sys.path.append(glob.glob('/external/carla/Dist/CARLA_Shipping_0.9.9.4-255-gc56a7738/LinuxNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import logging

import json

from PIL import Image
try:
    import queue
except ImportError:
    import Queue as queue


import subprocess
bashcmd = "getent hosts carla_server | awk '{ print $1 }'"
# server_ip = subprocess.check_output(['bash','-c',bashcmd])
# server_ip = str(server_ip.strip())
server_ip = 'localhost'

print("Found carla server at IP: ",server_ip)

from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def image_to_bgr_array(image, num_ch=4):
    # if not isinstance(image, carla.sensor.Image):
    #     raise ValueError("Argument must be a carla.sensor.Image")
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    return array

def semantic2motion(img, dynamic_labels):
    mask = np.zeros(img.shape[:-1])
    for _, label in dynamic_labels.items():
        mask = np.logical_or(mask, np.all(img == label, axis=-1))
    mask = mask.astype(np.uint8) * 255
    return mask

def process_img_rgb(image, file_name): #im_width, im_height, file_name):
    array = image_to_bgr_array(image)
    cv2.imwrite(file_name, array)


class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', 20)
        self._queues = []
        self._settings = None
        print("Sensors:",self.sensors)
    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


def should_quit(frame_cnt):
    return frame_cnt > 2500

print("Running for 1000 frames")

camera_origins = {
    # '0R': ([37.5, -59.69, 0.0], 90), 
    # '0L': ([112.5, -59.69, 0.0], 90), 
    # '1R': ([6.27, -103.69, 0.0], 135), 
    # '1L': ([59.31, -50.66, 0.0], 135), 
    # '2R': ([15.31, -156.88, 0.0], 180), 
    # '2L': ([15.31, -81.88, 0.0], 180), 
    # '3R': ([59.31, -188.11, 0.0], 225), 
    # '3L': ([6.27, -135.07, 0.0], 225), 
    # '4R': ([112.5, -179.07, 0.0], 270), 
    # '4L': ([37.5, -179.07, 0.0], 270), 
    # '5R': ([143.73, -135.07, 0.0], 315), 
    # '5L': ([90.69, -188.11, 0.0], 315), 
    '6R': ([134.69, -81.88, 0.0], 0), 
    '6L': ([134.69, -156.88, 0.0], 0), 
    # '7R': ([90.69, -50.66, 0.0], 45), 
    # '7L': ([143.73, -103.69, 0.0], 45)
    }

#BGR
dynamic_labels = {
    'pedestrian': np.array([142, 0, 0]),
    'vehicle': np.array([60, 20, 220])
}

params = {'fov': '82.55', 'image_size_x': '1242', 'image_size_y': '375', 'iso': '200', 'gamma': '2.2', 'shutter_speed': '30'} #400
rgb_positions = {k:(int(k[0])%4, int(int(k[0])/4) *2 + int(k[1]=='R')) for k in camera_origins.keys()}
other_positions = {k:(int(k[0])/4, int(k[0])%4) for k in camera_origins.keys() if "L" in k}
width = int(params['image_size_x']) #1920
height = int(params['image_size_y']) #1080

f = width / (2 * np.tan(float(params['fov']) * np.pi / 360))
cx = width / 2
cy = height / 2

intrinsics = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

def process_images_batch(image_tag_pairs, stereo=True, rgb=True):
    if rgb == 0:
        new_image = np.zeros((height*4, width*4, 3), np.uint8)
        for img_, tag in image_tag_pairs:
            i,j = rgb_positions[tag]
            img = image_to_bgr_array(img_)

            new_image[j*height: (j+1)*height, i*width: (i+1)*width] = img
    elif rgb == 1:
        new_image = np.zeros((height*4, width*2), np.float32)
        for img_, tag in image_tag_pairs:
            if "R" in tag:
                continue
            i,j = other_positions[tag]
            img = image_to_bgr_array(img_)
            img2 = img.astype(np.float32)
            img2 = np.dot(img2[:, :, :3], [65536.0, 256.0, 1.0])
            img2 /= 16777215.0
            new_image[j*height: (j+1)*height, i*width: (i+1)*width] = img2
    elif rgb == 2:
        new_image = np.zeros((height*4, width*2, 3), np.uint8)
        for img_, tag in image_tag_pairs:
            if "R" in tag:
                continue
            i,j = other_positions[tag]
            img_.convert(carla.ColorConverter.CityScapesPalette)
            img = image_to_bgr_array(img_)
            new_image[j*height: (j+1)*height, i*width: (i+1)*width] = img

    return new_image

def add_cameras(bp, vehicle, world):
    for k,v in params.items():
        if bp.has_attribute(k):
            bp.set_attribute(k, v)
    
    cc = carla.ColorConverter.Raw
    actor_list = []
    for k,v in camera_origins.items():
        camera_transform = carla.Transform(carla.Location(x=1.5+v[0][0]/1000, y=v[0][1]/1000, z=2.4), carla.Rotation(pitch=0,roll=0,yaw = v[1]))
        camera_ = world.spawn_actor(bp, camera_transform, attach_to=vehicle)

        actor_list.append((k,camera_))

    return actor_list

def main():
    actor_list = []

    client = carla.Client(server_ip, 2000)
    client.set_timeout(200000.0)
    
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_global_distance_to_leading_vehicle(2.0)

    synchronous_master = False

    walkers_list = []
    all_id = []
    number_of_walkers = 50

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    save_root_dir = '/external/datasets/carla_stereo/'
    world_name = 'Town03'

    np.savetxt(os.path.join(save_root_dir + world_name, 'intrinsics.txt'), intrinsics)

    try:
        print("Available maps: ", client.get_available_maps())
        try:
            world = client.load_world(world_name)
        except:
            world = client.get_world()
        print("using map:", world.get_map().name)
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_global_distance_to_leading_vehicle(2.0)

        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)
        else:
            synchronous_master = False

        blueprint_library = world.get_blueprint_library()
        blueprintsWalkers = world.get_blueprint_library().filter('walker.pedestrian.*')

        bp = random.choice(blueprint_library.filter('vehicle'))
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True, traffic_manager.get_port())
        vehicle.set_simulate_physics(False)

        # Let's add cameras attached to the vehicle. Note that the
        # transform we give here is now relative to the vehicle.

        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        print('at location ',transform.location)

        camera_bp = blueprint_library.find('sensor.camera.depth')
        camera_bp_rgb = blueprint_library.find('sensor.camera.rgb')
        camera_bp_ss = blueprint_library.find('sensor.camera.semantic_segmentation')

        cc = carla.ColorConverter.Raw #LogarithmicDepth
        rgb_cams = []
        depth_cams = []
        display_once = False

        rgb_cams_and_names = add_cameras(camera_bp_rgb, vehicle, world)
        depth_cams_and_names = add_cameras(camera_bp, vehicle, world)
        semantic_cams_and_names = add_cameras(camera_bp_ss, vehicle, world)

        rgb_cams = [rgbc[1] for rgbc in rgb_cams_and_names]
        rgb_names = [rgbc[0] for rgbc in rgb_cams_and_names]
        depth_cams = [dc[1] for dc in depth_cams_and_names]
        depth_names = [dc[0] for dc in depth_cams_and_names]
        semantic_cams = [ssc[1] for ssc in semantic_cams_and_names]
        semantic_names = [ssc[0] for ssc in semantic_cams_and_names]
        actor_list = actor_list + rgb_cams + depth_cams + semantic_cams

        cams_list = rgb_cams + depth_cams + semantic_cams

        location = vehicle.get_location()
        #print('Vehicle at %s' % location)

        # But the city now is probably quite empty, let's add a few more
        # vehicles.

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        for sp_pts in range(0, number_of_spawn_points):
            transform = spawn_points[sp_pts]
            bp = random.choice(blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, transform)
            if npc is not None:
                npc.set_autopilot(True, traffic_manager.get_port())
                actor_list.append(npc)
                #print('created %s' % npc.type_id)


        # -------------
        # Spawn Walkers
        # -------------
        # some settings


        SpawnActor = carla.command.SpawnActor
        # SetAutopilot = carla.command.SetAutopilot
        # SetVehicleLightState = carla.command.SetVehicleLightState
        # FutureActor = carla.command.FutureActor


        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                #print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

        cc_raw = carla.ColorConverter.LogarithmicDepth
        frame_cnt = 0

        # need to make sure world motion has begun and is stable
        stable = False

        # Create a synchronous mode context.
        with CarlaSyncMode(world, cams_list, fps=30) as sync_mode:
            while True:
                if should_quit(frame_cnt):
                    return
                
                output_images = sync_mode.tick(timeout=2.0)
                vehicle_snapshot = output_images[0].find(vehicle.id)

                if frame_cnt < 150 and stable is False:
                    print("stabilizing world: %i" % frame_cnt)
                    frame_cnt += 1
                    continue
                elif frame_cnt == 150 and stable is False:
                    frame_cnt = 0
                    stable = True

                #print("Vehicle location: ", vehicle_snapshot.get_transform().location)
                #print("Vehicle velocity: ", vehicle_snapshot.get_velocity())
                
                poses = {}

                # rgb_list = []
                for camera_idx, rgb_name in enumerate(rgb_names,1):
                    img = image_to_bgr_array(output_images[camera_idx]).astype(np.uint8)
                    if "L" in rgb_name:
                        cv2.imwrite(save_root_dir + world_name + '/left/rgb/rgb_%06d.png' % frame_cnt, img)
                    elif "R" in rgb_name:
                        cv2.imwrite(save_root_dir + world_name + '/right/rgb/rgb_%06d.png' % frame_cnt, img)
                    else: 
                        print("No frames")
                        sys.exit(0)
                    # rgb_list.append((output_images[camera_idx], rgb_name))
                    poses[rgb_name] = output_images[camera_idx].transform.get_matrix()
                # rgb_data = process_images_batch(rgb_list, rgb = 0)

                # depth_list = []
                for camera_idx, depth_name in enumerate(depth_names, 1+len(rgb_cams)):
                    img = image_to_bgr_array(output_images[camera_idx]).astype(np.float32)
                    img = np.dot(img[:, :, :3], [65536.0, 256.0, 1.0]) / 16777215.0
                    if "L" in depth_name:
                        Image.fromarray(img).save(save_root_dir + world_name + '/left/depth/depth_%06d.tif' % frame_cnt)
                    elif "R" in depth_name:
                        Image.fromarray(img).save(save_root_dir + world_name + '/right/depth/depth_%06d.tif' % frame_cnt)
                    else:
                        print("No frames")
                        sys.exit(0)
                    # depth_list.append((output_images[camera_idx], depth_name))
                # depth_data = process_images_batch(depth_list, rgb = 1)

                # semantic_list = []
                for camera_idx, semantic_name in enumerate(semantic_names, 1+len(rgb_cams)+len(depth_names)):
                    img = output_images[camera_idx]
                    img.convert(carla.ColorConverter.CityScapesPalette)
                    img = image_to_bgr_array(img)
                    img = semantic2motion(img, dynamic_labels).astype(np.uint8)
                    if "L" in semantic_name:
                        cv2.imwrite(save_root_dir + world_name + '/left/segmentation/segmentation_%06d.png' % frame_cnt, img)
                    elif "R" in semantic_name:
                        cv2.imwrite(save_root_dir + world_name + '/right/segmentation/segmentation_%06d.png' % frame_cnt, img)
                    else: 
                        print("No frames")
                        sys.exit(0)
                    # semantic_list.append((output_images[camera_idx], semantic_name))
                # semantic_data = process_images_batch(semantic_list, rgb = 2)

                # cv2.imwrite('./_out_stereo_test/rgb_%06d.png' % frame_cnt, rgb_data)
                # Image.fromarray(depth_data).save('./_out_stereo_test/depth_%06d.tif' % frame_cnt)
                # cv2.imwrite('./_out_stereo_test/semantic_%06d.png' % frame_cnt, semantic_data)

                print("Processed frame: %6d" % frame_cnt)

                acc = vehicle_snapshot.get_acceleration()
                omega = vehicle_snapshot.get_angular_velocity()
                vel = vehicle_snapshot.get_velocity()
                # T = vehicle_snapshot.get_transform()
                # loc = T.location
                # rot = T.rotation
                poses['vehicle'] = {
                    'acceleration' : np.asarray([acc.x, acc.y, acc.z], np.float64),
                    'angular_velocity' : np.asarray([omega.x, omega.y, omega.z], np.float64),
                    'transform' : vehicle_snapshot.get_transform().get_matrix(),
                    # 'Location' : np.asarray([loc.x, loc.y, loc.z], np.float64),
                    # 'yaw_degrees' : rot.yaw,
                    # 'pitch_degrees': rot.pitch,
                    # 'roll_degrees': rot.roll,
                    'velocity' : np.asarray([vel.x, vel.y, vel.z], np.float64)
                }
                #print("Poses: ",poses)
                with open(save_root_dir + world_name + '/poses/poses_%06d.json'%frame_cnt, 'w') as f_json:
                    json.dump(poses, f_json, cls = NumpyArrayEncoder, indent=4, sort_keys=True)
                    f_json.close()

                frame_cnt += 1
    except Exception as e:
        print("Caught something: ", e)
        print(sys.exc_info()[1])
        print(repr(e))

    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        print('done.')

if __name__ == '__main__':
    try:
        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
