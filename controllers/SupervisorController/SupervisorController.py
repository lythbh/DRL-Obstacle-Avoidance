from controller import Supervisor

sup = Supervisor()
timestep = int(sup.getBasicTimeStep())

# Altino node
altino_node = sup.getFromDef('ALTINO')
translation_field = altino_node.getField('translation')
rotation_field = altino_node.getField('rotation')
customData_field = altino_node.getField('customData')

start_position = [-5, 5, 0]
start_rotation = [0.0, 0.0, 1.0, 0.0]

while sup.step(timestep) != -1:
    customdata = customData_field.getSFString()

    # Reset position and pose 
    if customdata == 'reset':
        translation_field.setSFVec3f(start_position)
        rotation_field.setSFRotation(start_rotation)
        sup.simulationResetPhysics()
        customData_field.setSFString('')

    elif customdata == 'end simulation':
        sup.worldReload()