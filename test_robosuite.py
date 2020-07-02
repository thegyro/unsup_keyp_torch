import imageio
import numpy as np
import robosuite as suite

# create environment instance
#env = suite.make("SawyerLift", has_renderer=True)
import skimage
import skimage.transform

env = suite.make(
    "SawyerLift",
    has_renderer=True,          # no on-screen renderer
    has_offscreen_renderer=True, # off-screen renderer is required for camera observations
    ignore_done=True,            # (optional) never terminates episode
    use_camera_obs=True,         # use camera observations
    camera_height=500,            # set camera height
    camera_width=500,           # set camera width
    camera_name='sideview',     # use "agentview" camera
    use_object_obs=True,        # no object feature when training on pixels
    control_freq = 60)

# reset the environment
#env.viewer.set_camera(camera_id=2)
env.reset()

frames = []
for i in range(256):
    action = np.random.randn(env.dof)  # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    #print(env.dof, action.shape, obs)
    #env.render()  # render on display
    #print(obs.keys())
    print(obs['image'].shape, obs['depth'].shape)
    frames.append(skimage.img_as_ubyte(skimage.transform.rotate(obs['image'], 180)))
    #frames.append(skimage.img_as_ubyte(obs['image']))

imageio.mimsave("robosuite.mp4", frames, fps=30)