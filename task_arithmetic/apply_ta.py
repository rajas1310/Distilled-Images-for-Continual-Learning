from task_vectors import TaskVector
import torch
from net import ResNet


task_dict = {'cifar10' : {  0 : ['airplane', 'automobile'],
                            1 : ['bird', 'cat'],
                            2 : ['deer', 'dog'],
                            3 : ['frog', 'horse'],
                            4 : ['ship', 'truck']
                          },

             'mnist' : {    0 : [0, 9],
                            1 : [1, 8],
                            2 : [2, 7],
                            3 : [3, 6],
                            4 : [4, 5]
                        }
            }

# for i in range(6):
#     print(len(task_dict["oxfordpet"][i]))
# raise
ranges_of_classes = {
    "cifar10": {
        0: (0, 2),
        1: (2, 4),
        2: (4, 6),
        3: (6, 8),
        4: (8, 10)
    },

    "mnist": {
        0: (0, 2),
        1: (2, 4),
        2: (4, 6),
        3: (6, 8),
        4: (8, 10),
    }

}


def get_model(args, pretrained_checkpoint, list_of_task_checkpoints, scaling_coef):
    task_vector_list = [
        TaskVector(pretrained_checkpoint, task_checkpoint)
        for task_checkpoint in list_of_task_checkpoints
    ]
    vector_sum = sum(task_vector_list)
    model = vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    
    for param in model.parameters():
        param.requires_grad = False
    for task_idx, ckpt in enumerate(sorted(list_of_task_checkpoints)):
        finetuned_weights = torch.load(ckpt)
        # task_model = ViT_LoRA(args, use_)
        
        taskwise_model = ResNet(num_classes=args.num_classes, device=args.device, model=args.model)
        taskwise_model.load_state_dict(finetuned_weights)
        # print(taskwise_model.linear.weight.shape)
        start_idx = ranges_of_classes[args.dataset][task_idx][0]
        end_idx = ranges_of_classes[args.dataset][task_idx][1]
        model.linear.weight[start_idx:end_idx , :] = taskwise_model.linear.weight[start_idx:end_idx , :]
        model.linear.bias[start_idx:end_idx] = taskwise_model.linear.bias[start_idx:end_idx]
        # print(model.linear.weight.shape)
        # print(model.linear.bias.shape)
    return model