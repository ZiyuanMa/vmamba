# %%
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import v2
from mamba import MambaLMHeadModel, MambaConfig
from cnn import Model
from datasets import load_dataset
import config

input_dim = 64
num_layers = 8
batch_size = 256

dropout = 0.0
max_lr = 1e-3
wd = 0.05
path = './'
epoches = config.num_epoches
warmup_steps = 2000

transforms = v2.Compose([
    v2.PILToTensor(),
    v2.RandomResizedCrop(32, scale=(0.7, 1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
    # v2.Normalize(mean=[0.478], std=[0.268]),
    # v2.RandomErasing(),
    # v2.RandAugment(),
])

eval_transforms = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),

])
cutmix = v2.CutMix(num_classes=config.num_classes)
mixup = v2.MixUp(num_classes=config.num_classes)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
class DataSet(torch.utils.data.Dataset):
    def __init__(self, images, labels, trans):
        super().__init__()

        self.imgs = []
        self.labels = []
        self.trans = trans
        # for img, label in zip(data['image'], data['label']):
        for img, label in zip(images, labels):
            if len(np.array(img).shape) == 3:
                self.imgs.append(img)
                self.labels.append(label)

        # temp_image = [eval_transforms(img) for img in self.imgs]
        # temp_image = torch.stack(temp_image)
        # print(temp_image.size())
        # std0, mean0 = torch.std_mean(temp_image[:, 0])
        # std1, mean1 = torch.std_mean(temp_image[:, 1])
        # std2, mean2 = torch.std_mean(temp_image[:, 2])


        # print(std0)
        # print(mean0)
        # print(std1)
        # print(mean1)
        # print(std2)
        # print(mean2)
    def __getitem__(self, i):

        return self.trans(self.imgs[i]), self.labels[i]
    
    def __len__(self):
        return len(self.imgs)
    
def load_data(path):
    if config.dataset == 'cifar100':
        dataset = load_dataset("cifar100")
        train_data = dataset['train']
        valid_data = dataset['test']
        train_img, train_label = train_data['img'], train_data['fine_label']
        valid_img, valid_label = valid_data['img'], valid_data['fine_label']
    elif config.dataset == 'tiny-imagenet':
        dataset = load_dataset("zh-plus/tiny-imagenet")
        train_data = dataset['train']
        valid_data = dataset['valid']
        train_img, train_label = train_data['image'], train_data['label']
        valid_img, valid_label = valid_data['image'], valid_data['label']

    train_dataset = DataSet(train_img, train_label, transforms)
    valid_dataset = DataSet(valid_img, valid_label, eval_transforms)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

    total_train_steps = len(train_dataset) * epoches

    return train_dataloader, valid_dataloader, total_train_steps


# %%


train_dataloader, valid_dataloader, total_train_steps = load_data(path)



# %%


model = MambaLMHeadModel(MambaConfig(input_dim, num_layers)).cuda()
# model = Model(input_dim, num_layers).cuda()
# model = origin_model

    # model = origin_model
num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
print(f'number of param: {num_params}')
print(f'number of iters: {len(train_dataloader)}')

# %%


# %%

from tqdm import tqdm



def group_weight(module):
    group_decay = []
    group_no_decay = []
    for n, p in module.named_parameters():
        if 'norm' in n or 'bias' in n:
            group_no_decay.append(p)
        else:
            group_decay.append(p)
        
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups



weights = group_weight(model)
optimizer = torch.optim.AdamW(weights, lr=max_lr, weight_decay=wd)
# optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=wd, betas=(0.9, 0.98))

scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_train_steps-warmup_steps, eta_min=max_lr*0.01)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], [warmup_steps])

# %%
import math

from torch.utils.tensorboard import SummaryWriter

test_name = f"mamba_direction:{config.ssm_direction}_3_stages_rotate"
# test_name = 'cnn_4_stage'
writer = SummaryWriter(path+'/runs/'+test_name, max_queue=120)

scaler = torch.cuda.amp.GradScaler()

train_step = 0


best_loss = float('inf')
for i in range(epoches):

    losses = []
    
    model.train()
    for img, label in tqdm(train_dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):
        
        img = img.cuda()
        label = label.cuda()
        img, label = cutmix_or_mixup(img, label)
        batch_loss = 0
        model.train()
        # for mini_data in data.chunk(num_mini_batches, 0):
        with torch.autocast(device_type="cuda"):
            output = model(img)
            final_output = torch.reshape(output, (-1, output.shape[-1]))
            loss = F.cross_entropy(final_output, label, label_smoothing=0.1) 

        batch_loss += loss.item()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        scaler.step(optimizer)

        scaler.update()
 
        scheduler.step()
        optimizer.zero_grad()

        writer.add_scalar('Loss/train', batch_loss, train_step)
        losses.append(batch_loss)

        train_step += 1

    avg_loss = sum(losses)/len(losses)
    writer.add_scalar('Loss/train_average', avg_loss, i)


    
    losses.clear()
    accs = []
    model.eval()
    for img, label in tqdm(valid_dataloader, bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'):

        img = img.cuda()
        label = label.cuda()
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            
            output = model(img)  
            final_target = torch.reshape(label, (-1,))
            loss = F.cross_entropy(output, final_target, reduction='none')

            pred_labels = output.argmax(1)
            acc = (pred_labels==label).float().mean()
            accs.append(acc)


        losses.append(loss.cpu().numpy())
    
    avg_loss = np.mean(np.concatenate(losses, 0)).item()
    avg_acc= sum(accs) / len(accs)
    print(f'\n valid {i}: {avg_acc}')

    writer.add_scalar('Loss/valid', avg_loss, i)

    writer.add_scalar('Acc/valid', avg_acc, i)


    losses.clear()

    if avg_loss <= best_loss:
        print(avg_loss)
        best_loss = avg_loss
        torch.save(model.state_dict(), path+'/runs/'+test_name+'/best_model.pt')

