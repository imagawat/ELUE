import click
import os
import json
env_names = ['reach-v1', 'pick-place-v1', 'window-open-v1', 'basket-ball-v1', 'sweep-into-v1',  'dial-turn-v1']
# ['bin-picking-v1', 'box-close-v1', 'hand-insert-v1', 'door-lock-v1', 'door-unlock-v1', 'reach-v1', 'push-v1', 'pick-place-v1', 'reach-wall-v1', 'pick-place-wall-v1', 'push-wall-v1', 'door-open-v1', 'door-close-v1', 'drawer-open-v1', 'drawer-close-v1', 'button-press_topdown-v1', 'button-press-v1', 'button-press-topdown-wall-v1', 'button-press-wall-v1', 'peg-insert-side-v1', 'peg-unplug-side-v1', 'window-open-v1', 'window-close-v1', 'dissassemble-v1', 'hammer-v1', 'plate-slide-v1', 'plate-slide-side-v1', 'plate-slide-back-v1', 'plate-slide-back-side-v1', 'handle-press-v1', 'handle-pull-v1', 'handle-press-side-v1', 'handle-pull-side-v1', 'stick-push-v1', 'stick-pull-v1', 'basket-ball-v1', 'soccer-v1', 'faucet-open-v1', 'faucet-close-v1', 'coffee-push-v1', 'coffee-pull-v1', 'coffee-button-v1', 'sweep-v1', 'sweep-into-v1', 'pick-out-of-hole-v1', 'assembly-v1', 'shelf-place-v1', 'push-back-v1', 'lever-pull-v1', 'dial-turn-v1']

@click.command()
@click.option('--basename', default="")
@click.option('--suffix', default="")
def main(basename, suffix):
    with open(basename) as f:
        contents = json.load(f)
    for en in env_names:
        contents['env_name'] = en
        filename = en+'_'+suffix+'.json'
        with open(filename, 'w') as f:
            json.dump(contents, f)
        #output file
        for i in range(5):
            print("python generate_qsubcommand.py \"python -O launch_experiment.py --config configs/ml1/{}\" | sh".format(filename))
            #print("python generate_qsubcommand.py \"python -O launch_experiment.py --config configs/ml1/{} --use_debug_config \" | sh".format(filename))
if __name__ == "__main__":
    main()
