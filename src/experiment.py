import hydra
from procedure import Procedure
import os
import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


@hydra.main(config_path='../config/', config_name='defaults.yaml')
def main(config):
    with Procedure(config) as procedure:
        n_training_steps_done = 0
        n_iteration_collected = 0
        last_time_logged = -1e7
        while n_training_steps_done < config.n_training_steps:
            procedure.collect_data(config.n_samples_collected_per_training)
            n_iteration_collected += config.n_samples_collected_per_training
            loss = procedure.train(config.batch_size).numpy()
            n_training_steps_done += 1
            current_time = time.time()
            if current_time - last_time_logged > config.log_every:
                last_time_logged = current_time
                print(f"{n_training_steps_done=}  {n_iteration_collected=} {loss=}")
        print("Experiment finished, hope it worked. Good bye!")


if __name__ == "__main__":
    main()
