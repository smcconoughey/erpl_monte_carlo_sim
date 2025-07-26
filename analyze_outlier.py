import json
import numpy as np
import sys

dir_path = sys.argv[1]
sim_id = int(sys.argv[2])

with open(f'{dir_path}/sim_{sim_id}.json') as f:
    data = json.load(f)

velocity = np.array(data['velocity'])
angular_velocity = np.array(data['angular_velocity'])
altitude = np.array(data['altitude'])
speed = np.array(data['speed'])
euler_angles = np.array(data['euler_angles'])
quaternion = np.array(data['quaternion'])
stability_margin = np.array(data['stability_margin'])
print('Apogee:', data['apogee_altitude'])
print('Flight time:', data['flight_time'])
print('Max speed:', np.max(speed))
print('Final altitude:', altitude[-1])
print('Final velocity:', velocity[:, -1])
print('Initial attitude:', data['initial_conditions']['attitude'])
print('Stability margin min/max:', np.min(stability_margin), np.max(stability_margin))
print('Max |angular velocity|:', np.max(np.abs(angular_velocity)))
print('Has negative stability:', np.any(stability_margin < 0))
print('Min propellant fraction:', min(data['propellant_fraction']))
print('Has negative mass:', np.any(np.array(data['mass']) < 0))
print('Quaternion norms at start/end:', np.linalg.norm(quaternion[:, 0]), np.linalg.norm(quaternion[:, -1]))
print('Max quaternion norm deviation:', np.max(np.abs(np.linalg.norm(quaternion, axis=0) - 1)))
prop_frac = np.array(data['propellant_fraction'])
burnout_idx = np.argmax(prop_frac <= 0) if np.any(prop_frac <= 0) else len(prop_frac) - 1
print('Burnout index:', burnout_idx)
print('Burnout altitude:', altitude[burnout_idx])
print('Burnout speed:', speed[burnout_idx])
print('Burnout velocity:', velocity[:, burnout_idx])
print('Burnout quaternion:', quaternion[:, burnout_idx])
print('Burnout euler:', euler_angles[:, burnout_idx])
print('Burnout stability margin:', stability_margin[burnout_idx])

thrust = np.array(data['thrust'])
print('Max thrust after burnout:', np.max(thrust[burnout_idx:]))

speeds_post = speed[burnout_idx:]
if len(speeds_post) > 1:
    time_post = np.array(data['time'])[burnout_idx:]
    accel = np.diff(speeds_post) / np.diff(time_post)
    print('Max speed acceleration post burnout:', np.max(accel))
print('Min speed acceleration post burnout:', np.min(accel)) 