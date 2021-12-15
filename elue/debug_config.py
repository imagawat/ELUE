TRAINTEST=False #TODO:remove
IGNOREFILM=False #TODO:remove
IGNOREZ=False # ignore task information
PREDTEST=False #TODO:remove
MODIFYFILM=False #TODO:remove
EVALFROMSCRATCH=False # just using evaltask in training mode

# register
variables={'TRAINTEST': TRAINTEST, 'IGNOREFILM': IGNOREFILM, 'IGNOREZ': IGNOREZ, 'PREDTEST': PREDTEST, 'MODIFYFILM': MODIFYFILM, 'EVALFROMSCRATCH': EVALFROMSCRATCH}

# usage (from_scratch evaluation)
#EVALFROMSCRATCH=True
#IGNOREZ=True
#python generate_qsubcommand.py "python -O launch_experiment.py --config configs/ml1/reach-v1_SAC.json --use_debug_config" | sh

# usage (no embedding training)
#IGNOREZ=True
#python generate_qsubcommand.py "python -O launch_experiment.py --config configs/ml1/reach-v1_SAC.json --use_debug_config" | sh
