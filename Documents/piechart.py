#AUTHOR : VIKAS MELKOT BALASUBRAMANYAM
#MATRICULATION NUMBER : 66133
#Personal Programming Project
#=============================================================================

import matplotlib.pyplot as plt

Ts = [0.002, 0.702, 0.703, 0.704, 0.707, 0.707, 0.709, 0.710,
      0.713, 0.713, 0.717, 0.722, 0.727, 0.727, 24.391, 10.609, 10.612]

Total = sum(Ts)

Func_s = ['test_uniform_mesh', 'test_step_function', 'test_heaviside_functions', 'test_addAtPos',
          'test_connectivity_matrix', 'test_node_filtering', 'test_asymptotic_functions',
          'test_Gausspoints', 'test_tip_enrichment_func_N1', 'test_E_filter', 'test_LE_patch',
          'test_displacement_2x2', 'test_isotropic_material_prop', 'test_Jacobian', 'MainFunction',
          'test_Rigid_body_motions','test_Rigid_body_rotation']

# colors
colors = ['#FF0000', '#0000FF', '#FFA500', 'orangered', 'sienna', 'red', 'grey',
          'violet', 'r', 'b', 'maroon', 'orange', 'yellow', 'olive', 'g','navy', 'indigo']

# explosion
explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
           0.05, 0.05, 0.05, 0.05, 0.05)

fig, ax = plt.subplots(figsize=(20,20), subplot_kw=dict(aspect="equal"))
# Pie Chart
plt.pie(Ts, colors=colors, labels=Func_s,
        autopct='%1.1f%%', pctdistance=0.8,
        explode=explode, textprops={'fontsize': 18})

plt.legend(loc="center", fontsize=14)

# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()

# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

# Adding Title of chart
plt.title(f'Total time elapsed by all the functions is {Total} CPU seconds', fontsize=25)

# Displaying Chart
plt.show()
