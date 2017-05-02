import numpy as np

# implement returns calculation G_t for episode below: discount = 1.0
# practice traversing lists

def calculate_return(episodes, discount=1.0):
    #implement
    discounted_return = 0.0

    return discounted_return


episodes = [[(12, 4, False), 1, 0, (19, 4, False)], [(19, 4, False), 1, -1, (27, 4, False)], [(12, 4, False), 1, 0, (19, 4, False)], [(8, 6, False), 1, -1, (27, 4, False)] ]
calc_return = calculate_return(episodes)

expected_return = np.array([-2.0, -1.0, -2.0])
np.testing.assert_array_almost_equal(calc_return, expected_return, decimal=2)
