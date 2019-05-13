import random
import numpy as np

def corrupt_noise(traj, rate_noise, factor):
    new_traj={}
    for count, key in enumerate(traj):
        if count%500==0:
            print('count:',count)
        new_traj[key] = traj[key]
        for i in range(len(traj[key])):
                #adding gauss noise
                for col in range(46):
                    seed = random.random()
                    if seed < rate_noise:
                        new_traj[key][i][col] = traj[key][i][col] + factor * random.gauss(0,1)
    return new_traj

def corrupt_drop(traj, rate_drop):
    new_traj={}
    drop_id={}
    for count, key in enumerate(traj):
        if count%500==0:
            print('count:',count)
        new_traj[key] = traj[key]
        droprow = []
        for i in range(len(traj[key])):
            seed = random.random()
            if seed < rate_drop:
                #dropping
                droprow.append(i)
        new_traj[key] = np.delete(new_traj[key], droprow, axis = 0)
        drop_id[key]=droprow
    return drop_id #new_traj