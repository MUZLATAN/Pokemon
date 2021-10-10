import numpy
import torch
from PIL import Image
from torchvision import transforms
from ResNet import Resnet18
device = torch.device('cuda')
transform=transforms.Compose([
            transforms.Resize(280),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
                            ])

tf = transforms.Compose([
    lambda x: Image.open(x).convert('RGB'),  # 将t图片的路径转换可以处理图片数据
    # 进行数据加强
    transforms.Resize((int(224 * 1.25), int(224 * 1.25))),
    # 随机旋转
    # transforms.RandomRotation(15),  # 设置旋转的度数小一些，否则的话会增加网络的学习难度
    # 中心裁剪
    transforms.CenterCrop(224),  # 此时：既旋转了又不至于导致图片变得比较的复杂
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])


classes=['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']
def prediect(img_path):
    model = Resnet18(5).to(device) #模型初始化，5代表一共有5种类别
    model.load_state_dict(torch.load('model.pkl'))
    torch.no_grad()
    img=Image.open(img_path).convert('RGB')
    img=transform(img).unsqueeze(0)
    img_ = img.to(device)
    outputs = model(img_)

    print(outputs)
    _, predicted = torch.max(outputs, 1)
    # print(predicted)
    print('this picture maybe :',classes[predicted[0]])
if __name__ == '__main__':
    prediect('./test/chaomeng.jpg')

