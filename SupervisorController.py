from controller import Supervisor

sup = Supervisor()

TIMESTEP = 64

# Altino node
altino_node = sup.getFromDef('ALTINO')
# Position of Altino
translation_field = altino_node.getField('translation')

# Moving Altino loop
new_value = [0, 0, 0]
counter = 0
while sup.step(TIMESTEP) != -1:
    if counter % 100 == 0:
        translation_field.setSFVec3f(new_value)
    
    counter += 1 
