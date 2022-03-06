# from gym.envs.registration import register
# from my_envs.cartpole import CartpoleEnv
# from my_envs.vehicle import VehicleLateralEnv
# from my_envs.pendubot import PendubotEnv
# from my_envs.linearized_inverted_pendulum import LinearizedInvertedPendulumEnv
# from my_envs.inverted_pendulum import InvertedPendulumEnv

# # Cartpole
# register(
#     id = 'cartpole-full-v0',
#     entry_point = 'my_envs:CartpoleEnv',
#     kwargs = {
#         'observation': 'full',
#         'normed': False
#     }
# )
# register(
#     id = 'cartpole-partial-v0',
#     entry_point = 'my_envs:CartpoleEnv',
#     kwargs = {
#         'observation': 'partial',
#         'normed': False
#     }
# )
# register(
#     id = 'cartpole-partial-norm-v0',
#     entry_point = 'my_envs:CartpoleEnv',
#     kwargs = {
#         "observation": 'partial',
#         "normed": True
#     }
# )

# # Vehicle Lateral
# register(
#     id = 'vehicle-lateral-full-v0',
#     entry_point = 'my_envs:VehicleLateralEnv',
#     kwargs = {
#         'observation': 'full',
#         'normed': False
#     }
# )
# register(
#     id = 'vehicle-lateral-partial-v0',
#     entry_point = 'my_envs:VehicleLateralEnv',
#     kwargs = {
#         'observation': 'partial',
#         'normed': False
#     }
# )
# register(
#     id = 'vehicle-lateral-partial-norm-v0',
#     entry_point = 'my_envs:VehicleLateralEnv',
#     kwargs = {
#         "observation": 'partial',
#         "normed": True
#     }
# )

# # Pendubot
# register(
#     id = 'pendubot-full-v0',
#     entry_point = 'my_envs:PendubotEnv',
#     kwargs = {
#         'observation': 'full',
#         'normed': False
#     }
# )
# register(
#     id = 'pendubot-partial-v0',
#     entry_point = 'my_envs:PendubotEnv',
#     kwargs = {
#         'observation': 'partial',
#         'normed': False
#     }
# )
# register(
#     id = 'pendubot-partial-norm-v0',
#     entry_point = 'my_envs:PendubotEnv',
#     kwargs = {
#         "observation": 'partial',
#         "normed": True
#     }
# )

# # Linearized Inverted Pendulum
# register(
#     id = 'lin-inv-pendulum-full-v0',
#     entry_point = 'my_envs:LinearizedInvertedPendulumEnv',
#     kwargs = {
#         'observation': 'full',
#         'normed': False
#     }
# )
# register(
#     id = 'lin-inv-pendulum-partial-v0',
#     entry_point = 'my_envs:LinearizedInvertedPendulumEnv',
#     kwargs = {
#         'observation': 'partial',
#         'normed': False
#     }
# )
# register(
#     id = 'lin-inv-pendulum-partial-norm-v0',
#     entry_point = 'my_envs:LinearizedInvertedPendulumEnv',
#     kwargs = {
#         "observation": 'partial',
#         "normed": True
#     }
# )

# # Inverted Pendulum
# register(
#     id = 'inv-pendulum-full-v0',
#     entry_point = 'my_envs:InvertedPendulumEnv',
#     kwargs = {
#         'observation': 'full',
#         'normed': False
#     }
# )
# register(
#     id = 'inv-pendulum-partial-v0',
#     entry_point = 'my_envs:InvertedPendulumEnv',
#     kwargs = {
#         'observation': 'partial',
#         'normed': False
#     }
# )
# register(
#     id = 'inv-pendulum-partial-norm-v0',
#     entry_point = 'my_envs:InvertedPendulumEnv',
#     kwargs = {
#         "observation": 'partial',
#         "normed": True
#     }
# )