import numpy
import torch
import os,glob
import visdom
import time
import torchvision
import random,csv
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image

class Pokemon(Dataset):
    def __init__(self,root,resize,mode,):
        super(Pokemon,self).__init__()
        # 保存参数
        self.root=root
        self.resize=resize
        # 给每一个类做映射
        self.name2label={}  # "bulbasaur":0 ,"charmander":1 "mewtwo":2 "pikachu":3 "squirtle":4
        names = sorted(os.listdir(os.path.join(root)))
        for name in sorted(os.listdir(os.path.join(self.root))):  # listdir返回的顺序不固定，加上一个sorted使每一次的顺序都一样
            if not os.path.isdir(os.path.join(self.root, name)):  # os.path.isdir()用于判断括号中的内容是否是一个未压缩的文件夹
                continue
            self.name2label[name] = len(self.name2label.keys())
            
        print(self.name2label)
        # 加载文件
        self.images,self.labels=self.load_csv('images.csv')
        # 裁剪数据
        if mode=='train':
            self.images=self.images[:int(0.6*len(self.images))]   # 将数据集的60%设置为训练数据集合
            self.labels=self.labels[:int(0.6*len(self.labels))]   # label的60%分配给训练数据集合
        elif mode=='val':
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]  # 从60%-80%的地方
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]   # 从80%的地方到最末尾
            self.labels = self.labels[int(0.8 * len(self.labels)):]

        
        # image+label 的路径
    def load_csv(self,filename):
        # 将所有的图片加载进来
        # 如果不存在的话才进行创建
        if not os.path.exists(os.path.join(self.root, filename)):  # 如果filename这个文件不存在，那么执行以下代码，创建file
            images = []
            for name in self.name2label.keys():
                # glob.glob()返回的是括号中的路径中的所有文件的路径
                # += 是把glob.glob（）返回的结果依次append到image中，而不是以一个整体append
                # 这里只用了png/jpg/jepg是因为本次实验的图片只有这三种格式，如果有其他格式请自行添加
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            print(len(images))
            random.shuffle(images)  # 把所有图片路径顺序打乱
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:  # 将图片路径及其对应的数字标签写到指定文件中
                writer = csv.writer(f)
                for img in images:  # img e.g：'./pokemon/pikachu\\00000001.png'
                    name = img.split(os.sep)[-2]  # 即取出‘pikachu’
                    label = self.name2label[name]  # 根据name找到对应的数字标签
                    writer.writerow([img, label])  # 把每张图片的路径和它对应的数字标签写到指定的CSV文件中
                print('image paths and labels have been writen into csv file:', filename)

            # 把数据读入（如果filename存在就直接执行这一步，如果不存在就先创建file再读入数据）
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)  # 确保它们长度一致

        return images, labels

        # if not os.path.exists(os.path.join(self.root,filename)):
        #     images=[]
        #     for name in self.name2label.keys():
        #         images+=glob.glob(os.path.join(self.root,name,'*.png'))
        #         images+=glob.glob(os.path.join(self.root, name, '*.jpg'))
        #         images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
        #     print(len(images),images)
        #     # 1167 'pokeman\\bulbasaur\\00000000.png'
        #     # 这里是将所有文件夹下面的图片的名字读到images， 然后打乱顺序
        #     random.shuffle(images)
        #     with open(os.path.join(self.root,filename),mode='w',newline='') as f:
        #         writer=csv.writer(f)
        #         for img in images:    #  'pokeman\\bulbasaur\\00000000.png'
        #             #这里直接将采用os.sep  \\ 这个作为分隔符， 取倒数第二个 就是该图片的名字
        #             name=img.split(os.sep)[-2]
        #             #根据名字可以获得对应的图片的label
        #             label=self.name2label[name]
        #             #然后可以一行一行的写到.csv文件中
        #             writer.writerow([img,label])
        #         print("write into csv into :",filename)
        #
        # # 如果存在的话就直接的跳到这个地方
        # images,labels=[],[]
        # with open(os.path.join(self.root, filename)) as f:
        #     reader=csv.reader(f)
        #     for row in reader:
        #         # 接下来就会得到 'pokeman\\bulbasaur\\00000000.png' 0 的对象
        #         img,label=row
        #         y = numpy.asarray([0,0,0,0,0])
        #         y[int(label)] = 1
        #         images.append(img)
        #         labels.append(y)
        # # 保证images和labels的长度是一致的
        # assert len(images)==len(labels)
        # return images,labels

    # 返回数据的数量
    def __len__(self):
        return len(self.images)   # 返回的是被裁剪之后的关系

    def denormalize(self, x_hat):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean
        return x
    # 返回idx的数据和当前图片的label
    def __getitem__(self,idx):
        # idex-[0-总长度]
        # retrun images,labels
        # 将图片，label的路径取出来
        # 得到的img是这样的一个类型：'pokeman\\bulbasaur\\00000000.png'
        # 然而label得到的则是 0，1，2 这样的整形的格式
        img,label=self.images[idx],self.labels[idx]
        tf=transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),  # 将t图片的路径转换可以处理图片数据
            # 进行数据加强
            transforms.Resize((int(self.resize*1.25),int(self.resize*1.25))),
            # 随机旋转
            transforms.RandomRotation(15),   # 设置旋转的度数小一些，否则的话会增加网络的学习难度
            # 中心裁剪
            transforms.CenterCrop(self.resize),   # 此时：既旋转了又不至于导致图片变得比较的复杂
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])

        ])
        img=tf(img)
        label=torch.tensor(label)
        return img,label




def main():
    # 验证工作
    # viz=visdom.Visdom()

    db=Pokemon('./pokeman',64,'train')  # 这里可以改变大小 224->64,可以通过visdom进行查看
    # 可视化样本
    x,y=next(iter(db))
    print('sample:',x.shape,y.shape,y)
    # viz.image(db.denormalize(x),win='sample_x',opts=dict(title='sample_x'))
    # 加载batch_size的数据
    loader=DataLoader(db,batch_size=32,shuffle=True,num_workers=8)
    # for x,y in loader:
    #     viz.images(db.denormalize(x),nrow=8,win='batch',opts=dict(title='batch'))
    #     viz.text(str(y.numpy()),win='label',opts=dict(title='batch-y'))
    #     # 每一次加载后，休息10s
    #     time.sleep(10)

if __name__ == '__main__':
    main()