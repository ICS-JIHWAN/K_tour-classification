import wandb

from . import common

class WBLogger:
    def __init__(self, opt):
        self.batch_size = opt.batch_size
        wandb.init(project='tour', config=vars(opt))

    @staticmethod
    def log(prefix, metrics_dict):
        log_dict = {f'{prefix}_{key}': value[0] for key, value in metrics_dict.items()}
        wandb.log(log_dict)

    @staticmethod
    def log_tour_img_to_wandb(prefix, img, label, pred, step, opt):
        column_names = ['Image', 'Label', 'Prediction']

        output_table = wandb.Table(columns=column_names)
        for i in range(img.shape[0]):
            img_tmp = common.tensor2numpy(img[i])
            output_table.add_data(wandb.Image(img_tmp), label[i], pred[i])
        wandb.log({f'{prefix.title()} Step {step} Output Table': output_table})
