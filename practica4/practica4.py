import torch.nn as nn
import torch
import PIL
import torchvision
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F

class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = config.C + config.B*5
        layers = [
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for _ in range(4):
            layers.extend([
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1)
            ])
        
        layers.extend([
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])

        for _ in range(2):
            layers.extend([
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(0.1)
            ])
        
        layers.extend([
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1)
        ])

        layers.extend([
            nn.Flatten(),
            nn.Linear(1024*config.S*config.S, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, config.S*config.S*self.depth)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0), config.S, config.S, self.depth)
        )

class SumSquaredErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_coord = 5
        self.l_noobj = 0.5

    def forward(self, p, a):
        # Calculate IOU of each predicted bbox against the ground truth bbox
        iou = get_iou(p, a)                     # (batch, S, S, B, B)
        max_iou = torch.max(iou, dim=-1)[0]     # (batch, S, S, B)

        # Get masks
        bbox_mask = bbox_attr(a, 4) > 0.0
        p_template = bbox_attr(p, 4) > 0.0
        obj_i = bbox_mask[..., 0:1]         # 1 if grid I has any object at all
        responsible = torch.zeros_like(p_template).scatter_(       # (batch, S, S, B)
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),                # (batch, S, S, B)
            value=1                         # 1 if bounding box is "responsible" for predicting the object
        )
        obj_ij = obj_i * responsible        # 1 if object exists AND bbox is responsible
        noobj_ij = ~obj_ij                  # Otherwise, confidence should be 0

        # XY position losses
        x_losses = mse_loss(
            obj_ij * bbox_attr(p, 0),
            obj_ij * bbox_attr(a, 0)
        )
        y_losses = mse_loss(
            obj_ij * bbox_attr(p, 1),
            obj_ij * bbox_attr(a, 1)
        )
        pos_losses = x_losses + y_losses

        # Bbox dimension losses
        p_width = bbox_attr(p, 2)
        a_width = bbox_attr(a, 2)
        width_losses = mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + config.EPSILON),
            obj_ij * torch.sqrt(a_width)
        )
        p_height = bbox_attr(p, 3)
        a_height = bbox_attr(a, 3)
        height_losses = mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + config.EPSILON),
            obj_ij * torch.sqrt(a_height)
        )
        dim_losses = width_losses + height_losses

        # Confidence losses (target confidence is IOU)
        obj_confidence_losses = mse_loss(
            obj_ij * bbox_attr(p, 4),
            obj_ij * torch.ones_like(max_iou)
        )
        noobj_confidence_losses = mse_loss(
            noobj_ij * bbox_attr(p, 4),
            torch.zeros_like(max_iou)
        )

        # Classification losses
        class_losses = mse_loss(
            obj_i * p[..., :config.C],
            obj_i * a[..., :config.C]
        )

        total = self.l_coord * (pos_losses + dim_losses) \
                + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses
        return total / config.BATCH_SIZE


def mse_loss(a, b):
    flattened_a = torch.flatten(a, end_dim=-2)
    flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(
        flattened_a,
        flattened_b,
        reduction='sum'
    )


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, config.B, config.B, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),         # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(coords_join_size)          # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] \
                   * intersection_sides[..., 1]       # (batch, S, S, B, B)

    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)                  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(intersection)        # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)                  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(intersection)        # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = (union == 0.0)
    union[zero_unions] = config.EPSILON
    intersection[zero_unions] = 0.0

    return intersection / union


def bbox_attr(data, i):
    """Returns the Ith attribute of each bounding box in data."""

    attr_start = config.C + i
    return data[..., attr_start::5]


def bbox_to_coords(t):
    """Changes format of bounding boxes from [x, y, width, height] to ([x1, y1], [x2, y2])."""

    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)

def from_yolo_bounding_box(y_pred, S, B, C):

    y_pred = torch.reshape(y_pred, (y_pred.size(dim=0), S, S, C + B*5))
    
    box = torch.zeros((y_pred.size(dim=0), S*S*B, 4))
    prob_box = torch.zeros((y_pred.size(dim=0), S*S*B))
    prob = torch.zeros((y_pred.size(dim=0), S*S*B, C))
    for i in range(y_pred.size(dim=0)):
        for j in range(S):
            for k in range(S):
                for b in range(B):
                    cx, cy, w, h = y_pred[i, j, k, C + b*5:C + b*5 + 4].detach().numpy()
                    cx = (cx + j)/S
                    cy = (cy + k)/S
                    w = w**2
                    h = h**2
                    box[i, j*S + k*B + b] = torch.tensor([cx, cy, w, h])
                    prob_box[i, j*S + k*B + b] = y_pred[i, j, k, C + b*5 + 4]
                    prob[i, j*S + k*B + b] = y_pred[i, j, k, :C]
    return box, prob_box, prob

def to_yolo_bounding_box(bounding_boxes, classes, nboxes, grid_layout, image_size):

    C = len(classes)
    y_true = torch.zeros((len(bounding_boxes), grid_layout[0], grid_layout[1], C + nboxes*5))

    for i, raw in enumerate(bounding_boxes):
        for cell in raw:
            cx = (cell['xmin'] + cell['xmax'])/2.
            cy = (cell['ymin'] + cell['ymax'])/2.
            w = cell['xmax'] - cell['xmin']
            h = cell['ymax'] - cell['ymin']
            cx = cx / image_size[0] * grid_layout[0]
            cy = cy / image_size[1] * grid_layout[1]
            w = w / image_size[0]
            h = h / image_size[1]
            best_i = int(cx)
            best_j = int(cy)
            y_true[i, best_i, best_j, C + 4] = 1
            y_true[i, best_i, best_j, C] = cx - best_i
            y_true[i, best_i, best_j, C + 1] = cy - best_j
            y_true[i, best_i, best_j, C + 2] = w
            y_true[i, best_i, best_j, C + 3] = h
            y_true[i, best_i, best_j, :C + classes.index(cell['cell_type'])] = 1
    return y_true

class config:
    S = 7       # Divide each image into a SxS grid
    B = 2       # Number of bounding boxes to predict
    C = 3      # Number of classes in the dataset
    EPSILON = 1E-6
    BATCH_SIZE = 32


if __name__ == '__main__':

    images = os.listdir('./BCCD_Dataset/BCCD/')
    image_tensor = list()
    bounding_boxes = list()

    annotations = pd.read_csv('./BCCD_Dataset/annotations.csv')

    print(f'Number of images: {len(images)}')

    for image in images:
        dd = PIL.Image.open(f'./BCCD_Dataset/BCCD/{image}')
        tt = torchvision.transforms.functional.pil_to_tensor(dd)
        tt = tt.to(dtype=torch.float)

        xscaler = 448./tt.shape[2]
        yscaler = 448./tt.shape[1]

        tt = torchvision.transforms.functional.resize(tt, (448, 448))
        tt = tt[None, :, : :]

        image_tensor.append(tt)

        rows = annotations.loc[annotations['filename'] == image]
        rows = rows.reset_index()
        bounding_boxes.append([{'cell_type':rows['cell_type'][cellid],
                                'xmin': int(rows['xmin'][cellid] * xscaler),
                                'xmax': int(rows['xmax'][cellid] * xscaler),
                                'ymin': int(rows['ymin'][cellid] * yscaler), 
                                'ymax': int(rows['ymax'][cellid] * yscaler)} for cellid in range(len(rows))])

    image_tensor = torch.cat(image_tensor)

    print(f'Number of bounding boxes: {len(bounding_boxes)}')

    yolo_annotations = to_yolo_bounding_box(bounding_boxes, classes=['RBC', 'WBC', 'Platelets'], nboxes=2, grid_layout=(7, 7), image_size=(448, 448))

    yolo = YOLOv1()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo.to(device)
    yolo_loss = SumSquaredErrorLoss()
    optimizer = torch.optim.Adam(yolo.parameters(), lr=0.0001)
    
    dataloader_train_images = torch.utils.data.DataLoader(image_tensor, batch_size=10)
    dataloader_train_annotations = torch.utils.data.DataLoader(yolo_annotations, batch_size=10)

    loss_history = list()

    print('Training...')

    for epoch in range(20):
        for images, annotations in zip(dataloader_train_images, dataloader_train_annotations):
            yolo.train()
            optimizer.zero_grad()
            pred = yolo(images)
            loss = yolo_loss(pred, annotations)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        print(f'Epoch {epoch}: {loss_history[-1]}')

    plt.clf()
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig('./blood_cell/loss.png')

    print('Testing...')

    pred = yolo(image_tensor[:10])
    box, prob_box, prob = from_yolo_bounding_box(pred, 7, 2, 3)

    for i in range(len(box)):
        fig, ax = plt.subplots()
        ax.imshow(torch.permute(image_tensor[i], (1, 2, 0))/255.)
        for j in range(len(box[i])):
            cx, cy, w, h = box[i][j].detach().numpy()
            cx -= w/2.
            cy -= h/2.
            ax.add_patch(patches.Rectangle(xy=(cx, cy), width=w**2, height=h**2, alpha=0.3, color='b'))
            ax.text(cx, cy, f'{prob_box[i][j]:.1f}')

        for square in bounding_boxes[i]:
            cx = square['xmin']
            cy = square['ymin']
            w = square['xmax'] - square['xmin']
            h = square['ymax'] - square['ymin']
            ax.add_patch(patches.Rectangle(xy=(cx, cy), width=w, height=h, alpha=0.3, color='r'))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        fig.savefig(f'./blood_cell/yolo_{i}.png')
    plt.show()