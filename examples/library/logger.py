import datetime
from tensorflow import summary

class TensorboardLogger:
    def __init__(self,loc="./logs/dqn/",experiment="DQN",identifier=""):
        self.base_log_dir=loc
        self.experiment_name=experiment
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.folder_name=self.experiment_name + identifier + "-" +  current_time
        log_dir = self.base_log_dir + self.folder_name
        self.summary_writer = summary.create_file_writer(log_dir)

    def log(self,step,metrics):
        with self.summary_writer.as_default():
            for k,v in metrics.items():
                summary.scalar(f'{self.experiment_name}/{k}',v,step=step)